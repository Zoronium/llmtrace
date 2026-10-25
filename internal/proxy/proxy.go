// Package proxy implements a transparent tracing reverse proxy for
// OpenAI- and Anthropic-compatible LLM gateways.
package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Zoronium/llmtrace/internal/otlp"
)

// Config controls proxy behaviour.
type Config struct {
	Upstream           *url.URL
	CaptureContent     bool // send prompts and completions to the trace backend
	MaxCaptureBytes    int  // cap for captured input/output
	MaxRequestBytes    int64
	InjectUsage        bool // add stream_options.include_usage on OpenAI streams
	Environment        string
	ForwardTraceparent bool
}

// Proxy is an http.Handler.
type Proxy struct {
	cfg      Config
	rp       *httputil.ReverseProxy
	exporter otlp.Exporter
	log      *slog.Logger
	now      func() time.Time
}

type ctxKey struct{}

// operation describes a traced endpoint.
type operation struct {
	name      string // gen_ai.operation.name
	obsType   string // langfuse.observation.type
	openAIish bool   // supports stream_options.include_usage
}

var tracedOps = map[string]operation{
	"/chat/completions": {"chat", "generation", true},
	"/completions":      {"text_completion", "generation", true},
	"/messages":         {"chat", "generation", false},
	"/responses":        {"chat", "generation", false},
	"/embeddings":       {"embeddings", "embedding", false},
}

func classify(r *http.Request) (operation, string, bool) {
	if r.Method != http.MethodPost {
		return operation{}, "", false
	}
	p := strings.TrimRight(r.URL.Path, "/")
	for suffix, op := range tracedOps {
		if strings.HasSuffix(p, suffix) {
			return op, suffix, true
		}
	}
	return operation{}, "", false
}

// New builds a proxy.
func New(cfg Config, exp otlp.Exporter, log *slog.Logger) *Proxy {
	if cfg.MaxCaptureBytes <= 0 {
		cfg.MaxCaptureBytes = 32 << 10
	}
	if cfg.MaxRequestBytes <= 0 {
		cfg.MaxRequestBytes = 32 << 20
	}
	if exp == nil {
		exp = otlp.Noop{}
	}
	p := &Proxy{cfg: cfg, exporter: exp, log: log, now: time.Now}

	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   32,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: time.Second,
		ForceAttemptHTTP2:     true,
	}
	p.rp = &httputil.ReverseProxy{
		Rewrite:        p.rewrite,
		Transport:      transport,
		FlushInterval:  -1, // flush every write: never buffer SSE
		ModifyResponse: p.modifyResponse,
		ErrorHandler:   p.errorHandler,
	}
	return p
}

func (p *Proxy) rewrite(pr *httputil.ProxyRequest) {
	pr.SetURL(p.cfg.Upstream)
	pr.SetXForwarded()
	// Let Go's transport negotiate gzip and decompress transparently so the
	// tee parser always sees plain text.
	pr.Out.Header.Del("Accept-Encoding")
	for k := range pr.Out.Header {
		if strings.HasPrefix(strings.ToLower(k), "x-langfuse-") {
			pr.Out.Header.Del(k)
		}
	}
	if rec, ok := pr.In.Context().Value(ctxKey{}).(*record); ok && p.cfg.ForwardTraceparent {
		pr.Out.Header.Set("Traceparent", otlp.Traceparent(rec.traceID, rec.spanID))
	}
}

// ServeHTTP proxies every request; LLM calls are traced.
func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path == "/healthz" {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
		return
	}
	op, endpoint, traced := classify(r)
	if !traced {
		p.rp.ServeHTTP(w, r)
		return
	}

	start := p.now()
	body, err := io.ReadAll(io.LimitReader(r.Body, p.cfg.MaxRequestBytes+1))
	if err != nil {
		http.Error(w, "failed to read request body", http.StatusBadRequest)
		return
	}
	if int64(len(body)) > p.cfg.MaxRequestBytes {
		// Too large to trace safely: stream it through untouched.
		r.Body = io.NopCloser(io.MultiReader(bytes.NewReader(body), r.Body))
		p.rp.ServeHTTP(w, r)
		return
	}

	rec := &record{
		op:       op,
		endpoint: endpoint,
		start:    start,
		spanID:   otlp.NewSpanID(),
		res:      &result{},
	}
	if p.cfg.CaptureContent {
		rec.res.outputCap = p.cfg.MaxCaptureBytes
	}
	if tid, parent, ok := otlp.ParseTraceparent(r.Header.Get("Traceparent")); ok {
		rec.traceID, rec.parentID = tid, parent
	} else {
		rec.traceID = otlp.NewTraceID()
	}
	rec.sessionID = firstHeader(r, "X-Langfuse-Session-Id", "X-Session-Id")
	rec.userID = firstHeader(r, "X-Langfuse-User-Id", "X-User-Id")

	if m, info, ok := parseRequest(body); ok {
		rec.req = info
		if rec.userID == "" {
			rec.userID = info.User
		}
		if p.cfg.InjectUsage && op.openAIish && info.Stream && injectIncludeUsage(m) {
			if nb, err := json.Marshal(m); err == nil {
				body = nb
				rec.injectedUsage = true
			}
		}
	}

	r2 := r.Clone(context.WithValue(r.Context(), ctxKey{}, rec))
	r2.Body = io.NopCloser(bytes.NewReader(body))
	r2.ContentLength = int64(len(body))
	r2.Header.Set("Content-Length", itoa(int64(len(body))))

	// ReverseProxy copies the full response body before returning, so the
	// span can be finalized right after.
	p.rp.ServeHTTP(w, r2)
	rec.end = p.now()
	if r.Context().Err() != nil && rec.err == nil && !rec.done {
		rec.err = errors.New("client disconnected")
	}
	p.finish(rec)
}

func (p *Proxy) modifyResponse(resp *http.Response) error {
	rec, ok := resp.Request.Context().Value(ctxKey{}).(*record)
	if !ok {
		return nil
	}
	rec.status = resp.StatusCode
	rec.headersAt = p.now()
	rec.stream = strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream")
	rec.upstreamHeaders = interestingHeaders(resp.Header)
	resp.Body = &teeBody{rc: resp.Body, rec: rec, p: p}
	return nil
}

func (p *Proxy) errorHandler(w http.ResponseWriter, r *http.Request, err error) {
	if rec, ok := r.Context().Value(ctxKey{}).(*record); ok {
		rec.err = err
	}
	if errors.Is(err, context.Canceled) {
		return // client went away, nothing to write
	}
	p.log.Error("upstream error", "path", r.URL.Path, "err", err)
	w.WriteHeader(http.StatusBadGateway)
}

// record is the per-request state shared between the handler and the tee.
type record struct {
	op       operation
	endpoint string
	req      reqInfo

	traceID  otlp.TraceID
	spanID   otlp.SpanID
	parentID otlp.SpanID

	sessionID, userID string

	start, headersAt, end time.Time
	status                int
	stream                bool
	injectedUsage         bool
	upstreamHeaders       map[string]string

	res      *result
	raw      bytes.Buffer // non-stream body (bounded) or error body
	rawLimit bool
	sse      *sseParser
	bytesIn  int64
	done     bool // saw EOF
	err      error
}

const maxRawBody = 8 << 20

// teeBody observes the response body as ReverseProxy copies it to the client.
type teeBody struct {
	rc  io.ReadCloser
	rec *record
	p   *Proxy
}

func (t *teeBody) Read(b []byte) (int, error) {
	n, err := t.rc.Read(b)
	if n > 0 {
		t.observe(b[:n])
	}
	if errors.Is(err, io.EOF) {
		t.rec.done = true
	}
	return n, err
}

func (t *teeBody) Close() error { return t.rc.Close() }

func (t *teeBody) observe(b []byte) {
	rec := t.rec
	rec.bytesIn += int64(len(b))
	if rec.stream && rec.status < 400 {
		if rec.sse == nil {
			rec.sse = &sseParser{res: rec.res, now: t.p.now}
		}
		rec.sse.feed(b)
		return
	}
	if rec.raw.Len()+len(b) > maxRawBody {
		rec.rawLimit = true
		return
	}
	rec.raw.Write(b)
}

func (p *Proxy) finish(rec *record) {
	res := rec.res
	if !rec.stream && rec.status > 0 && rec.status < 400 && !rec.rawLimit {
		parseFull(rec.raw.Bytes(), res)
	}

	model := res.model
	if model == "" {
		model = rec.req.Model
	}
	dur := rec.end.Sub(rec.start)
	ms := func(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

	attrs := []otlp.Attr{
		otlp.String("langfuse.observation.type", rec.op.obsType),
		otlp.String("langfuse.trace.name", "llm "+strings.TrimPrefix(rec.endpoint, "/")),
		otlp.String("gen_ai.operation.name", rec.op.name),
		otlp.String("gen_ai.request.model", rec.req.Model),
		otlp.String("langfuse.observation.model.name", model),
		otlp.Bool("gen_ai.request.stream", rec.req.Stream),
		otlp.String("http.route", rec.endpoint),
		otlp.Int("http.response.status_code", int64(rec.status)),
		otlp.Int("llm.response.bytes", rec.bytesIn),
		otlp.Float("llm.duration_ms", ms(dur)),
		otlp.String("langfuse.observation.metadata.endpoint", rec.endpoint),
		otlp.Int("langfuse.observation.metadata.http_status", int64(rec.status)),
		otlp.Bool("langfuse.observation.metadata.stream", rec.stream),
	}
	if res.model != "" {
		attrs = append(attrs, otlp.String("gen_ai.response.model", res.model))
	}
	if res.provider != "" {
		attrs = append(attrs,
			otlp.String("gen_ai.provider.name", res.provider),
			otlp.String("langfuse.observation.metadata.provider", res.provider))
	}
	if p.cfg.Environment != "" {
		attrs = append(attrs, otlp.String("langfuse.environment", p.cfg.Environment))
	}
	if rec.sessionID != "" {
		attrs = append(attrs, otlp.String("langfuse.session.id", rec.sessionID))
	}
	if rec.userID != "" {
		attrs = append(attrs, otlp.String("langfuse.user.id", rec.userID))
	}
	if len(rec.req.Params) > 0 {
		if b, err := json.Marshal(rec.req.Params); err == nil {
			attrs = append(attrs, otlp.String("langfuse.observation.model.parameters", string(b)))
		}
	}
	if !rec.headersAt.IsZero() {
		ttfb := ms(rec.headersAt.Sub(rec.start))
		attrs = append(attrs,
			otlp.Float("llm.ttfb_ms", ttfb),
			otlp.Float("langfuse.observation.metadata.ttfb_ms", ttfb))
	}

	var ttft, tps float64
	if !res.firstToken.IsZero() {
		ttft = ms(res.firstToken.Sub(rec.start))
		attrs = append(attrs,
			otlp.String("langfuse.observation.completion_start_time", res.firstToken.UTC().Format(time.RFC3339Nano)),
			otlp.Float("llm.ttft_ms", ttft),
			otlp.Float("langfuse.observation.metadata.ttft_ms", ttft))
	}
	if res.hasUsage {
		details := map[string]int64{
			"input":  res.inputTokens,
			"output": res.outputTokens,
			"total":  res.inputTokens + res.outputTokens,
		}
		if res.cacheRead > 0 {
			details["input_cache_read"] = res.cacheRead
		}
		if res.cacheCreation > 0 {
			details["input_cache_creation"] = res.cacheCreation
		}
		b, _ := json.Marshal(details)
		attrs = append(attrs,
			otlp.String("langfuse.observation.usage_details", string(b)),
			otlp.Int("gen_ai.usage.input_tokens", res.inputTokens),
			otlp.Int("gen_ai.usage.output_tokens", res.outputTokens))

		// Decode throughput: output tokens over generation time (after first token
		// when streaming, whole request otherwise).
		gen := dur
		if !res.firstToken.IsZero() {
			gen = rec.end.Sub(res.firstToken)
		}
		if res.outputTokens > 0 && gen > 0 {
			tps = float64(res.outputTokens) / gen.Seconds()
			attrs = append(attrs,
				otlp.Float("llm.output_tokens_per_sec", tps),
				otlp.Float("langfuse.observation.metadata.output_tokens_per_sec", tps))
		}
	}
	if len(res.finish) > 0 {
		attrs = append(attrs, otlp.Strings("gen_ai.response.finish_reasons", res.finish))
	}

	// Delivery shape: distinguishes real token streaming from a gateway that
	// buffers the whole generation and replays it as a burst.
	st := res.stats()
	buffered := false
	if rec.stream && st.chunks > 0 {
		buffered = st.buffered(res.firstToken.Sub(rec.start), dur)
		attrs = append(attrs,
			otlp.Int("llm.stream.chunks", int64(st.chunks)),
			otlp.Float("llm.stream.generation_window_ms", ms(st.genWindow)),
			otlp.Float("llm.stream.max_gap_ms", ms(st.maxGap)),
			otlp.Float("llm.stream.median_gap_ms", ms(st.medianGap)),
			otlp.Bool("llm.stream.buffered", buffered),
			otlp.Bool("langfuse.observation.metadata.stream_buffered", buffered),
			otlp.Float("langfuse.observation.metadata.max_gap_ms", ms(st.maxGap)))
		if buffered && rec.err == nil && rec.status < 400 {
			attrs = append(attrs,
				otlp.String("langfuse.observation.level", "WARNING"),
				otlp.String("langfuse.observation.status_message", "stream appears buffered upstream: tokens arrived in a single burst"))
		}
	}
	if rec.injectedUsage {
		attrs = append(attrs, otlp.Bool("llm.proxy.injected_include_usage", true))
	}
	for k, v := range rec.upstreamHeaders {
		attrs = append(attrs, otlp.String("langfuse.observation.metadata.upstream."+k, v))
	}

	if p.cfg.CaptureContent {
		if in := captureInput(rec.req, p.cfg.MaxCaptureBytes); in != "" {
			attrs = append(attrs, otlp.String("langfuse.observation.input", in))
		}
		if res.output.Len() > 0 {
			out, _ := json.Marshal(res.output.String())
			attrs = append(attrs, otlp.String("langfuse.observation.output", string(out)))
			if res.truncated {
				attrs = append(attrs, otlp.Bool("llm.output_truncated", true))
			}
		}
	}

	status, msg := otlp.StatusOK, ""
	switch {
	case rec.err != nil:
		status, msg = otlp.StatusError, rec.err.Error()
	case rec.status >= 400:
		status, msg = otlp.StatusError, errorSummary(rec.status, rec.raw.Bytes())
	case rec.status == 0:
		status, msg = otlp.StatusError, "no response from upstream"
	}
	if status == otlp.StatusError {
		attrs = append(attrs,
			otlp.String("langfuse.observation.level", "ERROR"),
			otlp.String("langfuse.observation.status_message", msg))
	}

	p.exporter.Export(otlp.Span{
		TraceID:       rec.traceID,
		SpanID:        rec.spanID,
		ParentSpanID:  rec.parentID,
		Name:          rec.op.name + " " + model,
		Kind:          otlp.KindClient,
		Start:         rec.start,
		End:           rec.end,
		Attrs:         attrs,
		StatusCode:    status,
		StatusMessage: msg,
	})

	// One structured line per request: easy to grep, and the raw material for
	// aggregate stats later.
	p.log.Info("llm_request",
		"trace_id", rec.traceID.String(),
		"endpoint", rec.endpoint,
		"model_req", rec.req.Model,
		"model_resp", res.model,
		"stream", rec.stream,
		"status", rec.status,
		"duration_ms", round(ms(dur)),
		"ttft_ms", round(ttft),
		"tokens_in", res.inputTokens,
		"tokens_out", res.outputTokens,
		"tps", round(tps),
		"provider", res.provider,
		"chunks", st.chunks,
		"gen_window_ms", round(ms(st.genWindow)),
		"max_gap_ms", round(ms(st.maxGap)),
		"median_gap_ms", round(ms(st.medianGap)),
		"buffered", buffered,
		"error", msg,
	)
}

func captureInput(req reqInfo, limit int) string {
	if len(req.Input) == 0 && len(req.System) == 0 {
		return ""
	}
	var v any
	if len(req.System) > 0 {
		v = map[string]json.RawMessage{"system": req.System, "messages": req.Input}
	} else {
		v = req.Input
	}
	b, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	if len(b) > limit {
		wrapped, _ := json.Marshal(map[string]any{
			"truncated":      true,
			"original_bytes": len(b),
			"preview":        string(b[:limit]),
		})
		return string(wrapped)
	}
	return string(b)
}

func errorSummary(status int, body []byte) string {
	s := strings.TrimSpace(string(body))
	if len(s) > 500 {
		s = s[:500] + "..."
	}
	if s == "" {
		return http.StatusText(status)
	}
	return s
}

// interestingHeaders keeps x-* response headers (provider hints, rate limits)
// while dropping anything that looks like a credential.
func interestingHeaders(h http.Header) map[string]string {
	out := map[string]string{}
	keys := make([]string, 0, len(h))
	for k := range h {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		lk := strings.ToLower(k)
		if !strings.HasPrefix(lk, "x-") || lk == "x-forwarded-for" {
			continue
		}
		if strings.Contains(lk, "key") || strings.Contains(lk, "token") ||
			strings.Contains(lk, "auth") || strings.Contains(lk, "cookie") {
			continue
		}
		v := h.Get(k)
		if len(v) > 200 {
			v = v[:200]
		}
		out[lk] = v
		if len(out) >= 30 {
			break
		}
	}
	return out
}

func firstHeader(r *http.Request, names ...string) string {
	for _, n := range names {
		if v := r.Header.Get(n); v != "" {
			return v
		}
	}
	return ""
}

func round(f float64) float64 { return float64(int64(f*10+0.5)) / 10 }

func itoa(n int64) string { return strconv.FormatInt(n, 10) }
