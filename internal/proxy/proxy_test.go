package proxy

import (
	"bufio"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Zoronium/llmtrace/internal/otlp"
)

type captureExporter struct {
	mu    sync.Mutex
	spans []otlp.Span
}

func (c *captureExporter) Export(s otlp.Span) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.spans = append(c.spans, s)
}
func (c *captureExporter) Shutdown(context.Context) error { return nil }
func (c *captureExporter) all() []otlp.Span {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]otlp.Span(nil), c.spans...)
}

func attrs(s otlp.Span) map[string]any {
	m := map[string]any{}
	for _, a := range s.Attrs {
		m[a.Key] = a.Value
	}
	return m
}

func setup(t *testing.T, upstream http.Handler, capture bool) (*httptest.Server, *captureExporter) {
	t.Helper()
	up := httptest.NewServer(upstream)
	t.Cleanup(up.Close)
	u, _ := url.Parse(up.URL)
	exp := &captureExporter{}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	p := New(Config{
		Upstream:           u,
		CaptureContent:     capture,
		InjectUsage:        true,
		ForwardTraceparent: true,
	}, exp, log)
	srv := httptest.NewServer(p)
	t.Cleanup(srv.Close)
	return srv, exp
}

func waitSpans(t *testing.T, exp *captureExporter, n int) []otlp.Span {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if s := exp.all(); len(s) >= n {
			return s
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("expected %d spans, got %d", n, len(exp.all()))
	return nil
}

func sse(w http.ResponseWriter, payload string) {
	fmt.Fprintf(w, "data: %s\n\n", payload)
	w.(http.Flusher).Flush()
}

// The core promise: chunks reach the client as they are produced, and the span
// carries TTFT, usage and throughput.
func TestOpenAIStreamPassthroughAndTiming(t *testing.T) {
	var gotIncludeUsage bool
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		if so, ok := body["stream_options"].(map[string]any); ok {
			gotIncludeUsage, _ = so["include_usage"].(bool)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("X-Provider", "groq")
		w.Header().Set("X-Api-Key-Hint", "secret")
		sse(w, `{"model":"llama-3.3-70b","choices":[{"delta":{"role":"assistant"}}]}`)
		time.Sleep(50 * time.Millisecond)
		sse(w, `{"model":"llama-3.3-70b","choices":[{"delta":{"content":"Hello"}}]}`)
		time.Sleep(300 * time.Millisecond) // a slow tail must not delay the first token
		sse(w, `{"model":"llama-3.3-70b","choices":[{"delta":{"content":" world"},"finish_reason":"stop"}]}`)
		sse(w, `{"model":"llama-3.3-70b","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":40,"total_tokens":52}}`)
		sse(w, `[DONE]`)
	})
	srv, exp := setup(t, upstream, true)

	start := time.Now()
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"auto","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()

	br := bufio.NewReader(resp.Body)
	var firstContentAt time.Duration
	var all strings.Builder
	for {
		line, err := br.ReadString('\n')
		all.WriteString(line)
		if firstContentAt == 0 && strings.Contains(line, `"Hello"`) {
			firstContentAt = time.Since(start)
		}
		if err != nil {
			break
		}
	}
	if !gotIncludeUsage {
		t.Error("proxy did not inject stream_options.include_usage")
	}
	if firstContentAt == 0 || firstContentAt > 250*time.Millisecond {
		t.Errorf("first token reached client after %v: stream is being buffered", firstContentAt)
	}
	if !strings.Contains(all.String(), "[DONE]") {
		t.Error("client did not receive the full stream")
	}

	s := waitSpans(t, exp, 1)[0]
	a := attrs(s)
	if a["langfuse.observation.type"] != "generation" {
		t.Errorf("type = %v", a["langfuse.observation.type"])
	}
	if a["gen_ai.response.model"] != "llama-3.3-70b" || a["gen_ai.request.model"] != "auto" {
		t.Errorf("models: req=%v resp=%v", a["gen_ai.request.model"], a["gen_ai.response.model"])
	}
	if a["gen_ai.usage.input_tokens"] != int64(12) || a["gen_ai.usage.output_tokens"] != int64(40) {
		t.Errorf("usage: %v / %v", a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"])
	}
	ttft, _ := a["llm.ttft_ms"].(float64)
	dur, _ := a["llm.duration_ms"].(float64)
	if ttft < 40 || ttft > 250 || dur < 330 {
		t.Errorf("timings off: ttft=%vms duration=%vms", ttft, dur)
	}
	if _, ok := a["langfuse.observation.completion_start_time"]; !ok {
		t.Error("missing completion_start_time")
	}
	if tps, _ := a["llm.output_tokens_per_sec"].(float64); tps <= 0 {
		t.Errorf("tps = %v", tps)
	}
	if a["langfuse.observation.output"] != `"Hello world"` {
		t.Errorf("output = %v", a["langfuse.observation.output"])
	}
	if a["langfuse.observation.metadata.upstream.x-provider"] != "groq" {
		t.Error("provider header not captured")
	}
	if _, ok := a["langfuse.observation.metadata.upstream.x-api-key-hint"]; ok {
		t.Error("credential-looking header leaked into span")
	}
	if s.StatusCode != otlp.StatusOK {
		t.Errorf("status = %d", s.StatusCode)
	}
}

// Reproduces what we saw on OpenRouter free models: a long silence, then the
// whole generation in one burst. The proxy should flag it and keep the provider.
func TestBufferedStreamDetection(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		time.Sleep(300 * time.Millisecond)
		var b strings.Builder
		for i := 0; i < 15; i++ {
			fmt.Fprintf(&b, "data: {\"model\":\"dots:free\",\"provider\":\"AtlasCloud\",\"choices\":[{\"delta\":{\"content\":\"t%d \"}}]}\n\n", i)
		}
		b.WriteString("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":15}}\n\ndata: [DONE]\n\n")
		_, _ = w.Write([]byte(b.String())) // one write: everything arrives together
		w.(http.Flusher).Flush()
	})
	srv, exp := setup(t, upstream, false)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"auto","stream":true,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	a := attrs(waitSpans(t, exp, 1)[0])
	if a["llm.stream.buffered"] != true {
		t.Errorf("burst not detected: chunks=%v median=%v", a["llm.stream.chunks"], a["llm.stream.median_gap_ms"])
	}
	if a["langfuse.observation.level"] != "WARNING" {
		t.Errorf("level = %v", a["langfuse.observation.level"])
	}
	if a["gen_ai.provider.name"] != "AtlasCloud" || a["langfuse.observation.metadata.provider"] != "AtlasCloud" {
		t.Errorf("provider = %v", a["gen_ai.provider.name"])
	}
	if a["llm.stream.chunks"] != int64(15) {
		t.Errorf("chunks = %v", a["llm.stream.chunks"])
	}
}

// A genuinely streamed response with steady gaps must not be flagged.
func TestSteadyStreamNotFlagged(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 12; i++ {
			sse(w, fmt.Sprintf(`{"model":"m","choices":[{"delta":{"content":"t%d "}}]}`, i))
			time.Sleep(10 * time.Millisecond)
		}
		sse(w, `[DONE]`)
	})
	srv, exp := setup(t, upstream, false)
	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"m","stream":true,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	a := attrs(waitSpans(t, exp, 1)[0])
	if a["llm.stream.buffered"] != false {
		t.Errorf("steady stream flagged as buffered")
	}
	if g, _ := a["llm.stream.median_gap_ms"].(float64); g < 5 {
		t.Errorf("median gap = %vms, expected ~10ms", g)
	}
	if _, ok := a["langfuse.observation.level"]; ok {
		t.Errorf("unexpected level %v", a["langfuse.observation.level"])
	}
}

func TestAnthropicStream(t *testing.T) {
	var rawBody string
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		rawBody = string(b)
		w.Header().Set("Content-Type", "text/event-stream")
		write := func(event, data string) {
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
			w.(http.Flusher).Flush()
		}
		write("message_start", `{"type":"message_start","message":{"model":"claude-x","usage":{"input_tokens":25,"output_tokens":1}}}`)
		write("content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		time.Sleep(20 * time.Millisecond)
		write("content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Namaste"}}`)
		write("message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}`)
		write("message_stop", `{"type":"message_stop"}`)
	})
	srv, exp := setup(t, upstream, true)

	body := `{"model":"claude-x","stream":true,"max_tokens":100,"system":"be brief","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(srv.URL+"/v1/messages", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	_, _ = io.Copy(io.Discard, resp.Body)
	resp.Body.Close()

	if strings.Contains(rawBody, "stream_options") {
		t.Error("stream_options must not be injected into Anthropic requests")
	}
	a := attrs(waitSpans(t, exp, 1)[0])
	if a["gen_ai.usage.input_tokens"] != int64(25) || a["gen_ai.usage.output_tokens"] != int64(7) {
		t.Errorf("usage: %v / %v", a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"])
	}
	if fr, _ := a["gen_ai.response.finish_reasons"].([]string); len(fr) != 1 || fr[0] != "end_turn" {
		t.Errorf("finish reasons = %v", a["gen_ai.response.finish_reasons"])
	}
	if a["langfuse.observation.output"] != `"Namaste"` {
		t.Errorf("output = %v", a["langfuse.observation.output"])
	}
	in, _ := a["langfuse.observation.input"].(string)
	if !strings.Contains(in, "be brief") {
		t.Errorf("system prompt missing from input: %s", in)
	}
	if mp, _ := a["langfuse.observation.model.parameters"].(string); !strings.Contains(mp, "max_tokens") {
		t.Errorf("model parameters = %v", mp)
	}
}

func TestNonStreamJSONGzipped(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		payload := `{"model":"gemini-2.5-flash","choices":[{"message":{"role":"assistant","content":"42"},"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":3}}`
		w.Header().Set("Content-Type", "application/json")
		if strings.Contains(r.Header.Get("Accept-Encoding"), "gzip") {
			w.Header().Set("Content-Encoding", "gzip")
			gz := gzip.NewWriter(w)
			_, _ = gz.Write([]byte(payload))
			_ = gz.Close()
			return
		}
		_, _ = w.Write([]byte(payload))
	})
	srv, exp := setup(t, upstream, false)

	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions",
		strings.NewReader(`{"model":"auto","messages":[{"role":"user","content":"meaning of life"}]}`))
	req.Header.Set("X-Session-Id", "sess-1")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if !strings.Contains(string(got), `"42"`) {
		t.Fatalf("client body corrupted: %q", got)
	}

	a := attrs(waitSpans(t, exp, 1)[0])
	if a["gen_ai.usage.output_tokens"] != int64(3) || a["gen_ai.response.model"] != "gemini-2.5-flash" {
		t.Errorf("parse failed: %v", a)
	}
	if a["langfuse.session.id"] != "sess-1" {
		t.Errorf("session = %v", a["langfuse.session.id"])
	}
	if _, ok := a["langfuse.observation.input"]; ok {
		t.Error("content captured although CAPTURE_CONTENT is off")
	}
	if _, ok := a["llm.ttft_ms"]; ok {
		t.Error("non-stream request should not report TTFT")
	}
}

func TestUpstreamRateLimit(t *testing.T) {
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded for model"}}`))
	})
	srv, exp := setup(t, upstream, false)

	resp, err := http.Post(srv.URL+"/v1/chat/completions", "application/json",
		strings.NewReader(`{"model":"auto","stream":true,"messages":[]}`))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != 429 || !strings.Contains(string(b), "rate limit") {
		t.Fatalf("client got %d %q", resp.StatusCode, b)
	}
	s := waitSpans(t, exp, 1)[0]
	a := attrs(s)
	if s.StatusCode != otlp.StatusError || a["langfuse.observation.level"] != "ERROR" {
		t.Errorf("status=%d level=%v", s.StatusCode, a["langfuse.observation.level"])
	}
	if msg, _ := a["langfuse.observation.status_message"].(string); !strings.Contains(msg, "rate limit") {
		t.Errorf("status message = %q", msg)
	}
}

func TestTraceparentPropagationAndPassthrough(t *testing.T) {
	var gotTP string
	upstream := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotTP = r.Header.Get("Traceparent")
		if r.URL.Path == "/v1/models" {
			_, _ = w.Write([]byte(`{"data":[]}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","usage":{"prompt_tokens":1,"completion_tokens":1},"choices":[]}`))
	})
	srv, exp := setup(t, upstream, false)

	// Untraced endpoint passes through without a span.
	resp, err := http.Get(srv.URL + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	parent := "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
	req, _ := http.NewRequest("POST", srv.URL+"/v1/chat/completions", strings.NewReader(`{"model":"m"}`))
	req.Header.Set("Traceparent", parent)
	resp, err = http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()

	spans := waitSpans(t, exp, 1)
	time.Sleep(50 * time.Millisecond)
	if n := len(exp.all()); n != 1 {
		t.Fatalf("expected exactly 1 span (models endpoint untraced), got %d", n)
	}
	s := spans[0]
	if s.TraceID.String() != "4bf92f3577b34da6a3ce929d0e0e4736" || s.ParentSpanID.String() != "00f067aa0ba902b7" {
		t.Errorf("parent not honoured: trace=%s parent=%s", s.TraceID, s.ParentSpanID)
	}
	if !strings.Contains(gotTP, "4bf92f3577b34da6a3ce929d0e0e4736") || !strings.Contains(gotTP, s.SpanID.String()) {
		t.Errorf("upstream traceparent = %q", gotTP)
	}
}
