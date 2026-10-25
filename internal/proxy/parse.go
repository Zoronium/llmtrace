package proxy

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"
	"time"
)

// usage covers the token-count shapes of OpenAI chat/completions, the
// Responses API, and Anthropic messages.
type usage struct {
	PromptTokens        int64 `json:"prompt_tokens"`
	CompletionTokens    int64 `json:"completion_tokens"`
	InputTokens         int64 `json:"input_tokens"`
	OutputTokens        int64 `json:"output_tokens"`
	CacheReadTokens     int64 `json:"cache_read_input_tokens"`
	CacheCreationTokens int64 `json:"cache_creation_input_tokens"`
}

func (u *usage) input() int64 {
	if u.PromptTokens > 0 {
		return u.PromptTokens
	}
	return u.InputTokens
}

func (u *usage) output() int64 {
	if u.CompletionTokens > 0 {
		return u.CompletionTokens
	}
	return u.OutputTokens
}

// result accumulates everything we learn from the upstream response body.
type result struct {
	model         string
	inputTokens   int64
	outputTokens  int64
	cacheRead     int64
	cacheCreation int64
	hasUsage      bool
	finish        []string
	firstToken    time.Time
	lastToken     time.Time
	tokenEvents   int
	gaps          []time.Duration // time between consecutive token-bearing events
	provider      string          // upstream provider when the gateway reports it (OpenRouter-style "provider")
	events        int

	output    strings.Builder
	outputCap int
	truncated bool
}

func (r *result) addUsage(u *usage) {
	if u == nil {
		return
	}
	// Merge rather than overwrite: Anthropic sends input tokens in
	// message_start and output tokens later in message_delta.
	if v := u.input(); v > 0 {
		r.inputTokens = v
		r.hasUsage = true
	}
	if v := u.output(); v > 0 {
		r.outputTokens = v
		r.hasUsage = true
	}
	if u.CacheReadTokens > 0 {
		r.cacheRead = u.CacheReadTokens
	}
	if u.CacheCreationTokens > 0 {
		r.cacheCreation = u.CacheCreationTokens
	}
}

func (r *result) addFinish(reason string) {
	if reason == "" {
		return
	}
	for _, f := range r.finish {
		if f == reason {
			return
		}
	}
	r.finish = append(r.finish, reason)
}

func (r *result) appendText(s string) {
	if s == "" || r.outputCap <= 0 {
		return
	}
	room := r.outputCap - r.output.Len()
	if room <= 0 {
		r.truncated = true
		return
	}
	if len(s) > room {
		s = s[:room]
		r.truncated = true
	}
	r.output.WriteString(s)
}

const maxGaps = 50_000

// markToken records the arrival of a token-bearing event. Arrival time is when
// the bytes reached the proxy, so events that arrive in the same read share a
// timestamp: a run of zero gaps means upstream buffered and then burst.
func (r *result) markToken(now time.Time) {
	if r.firstToken.IsZero() {
		r.firstToken = now
	} else if len(r.gaps) < maxGaps {
		r.gaps = append(r.gaps, now.Sub(r.lastToken))
	}
	r.lastToken = now
	r.tokenEvents++
}

// streamStats summarises how output was delivered.
type streamStats struct {
	chunks    int
	genWindow time.Duration // first token to last token
	maxGap    time.Duration
	medianGap time.Duration
}

func (r *result) stats() streamStats {
	s := streamStats{chunks: r.tokenEvents}
	if r.tokenEvents > 0 {
		s.genWindow = r.lastToken.Sub(r.firstToken)
	}
	if len(r.gaps) == 0 {
		return s
	}
	sorted := append([]time.Duration(nil), r.gaps...)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	s.maxGap = sorted[len(sorted)-1]
	s.medianGap = sorted[len(sorted)/2]
	return s
}

// buffered reports whether a stream looks like it was produced in full upstream
// and released in a burst: most of the wall time is spent before the first
// token, then the remaining chunks arrive back to back.
func (s streamStats) buffered(ttft, total time.Duration) bool {
	if s.chunks < 10 || total <= 0 {
		return false
	}
	return s.medianGap < time.Millisecond && float64(ttft)/float64(total) > 0.5
}

// ---- streaming (SSE) ----

// sseParser incrementally splits an SSE byte stream into data payloads.
type sseParser struct {
	pending []byte
	res     *result
	now     func() time.Time
}

const maxPendingLine = 1 << 20 // 1 MiB without a newline means something is wrong

func (p *sseParser) feed(b []byte) {
	p.pending = append(p.pending, b...)
	for {
		i := bytes.IndexByte(p.pending, '\n')
		if i < 0 {
			break
		}
		line := bytes.TrimRight(p.pending[:i], "\r")
		p.pending = p.pending[i+1:]
		p.line(line)
	}
	if len(p.pending) > maxPendingLine {
		p.pending = p.pending[:0]
	}
	// Reclaim memory once the buffer drains.
	if len(p.pending) == 0 && cap(p.pending) > 64<<10 {
		p.pending = nil
	}
}

func (p *sseParser) line(line []byte) {
	if !bytes.HasPrefix(line, []byte("data:")) {
		return // event:, id:, retry:, comments and blank separators
	}
	payload := bytes.TrimSpace(line[len("data:"):])
	if len(payload) == 0 || bytes.Equal(payload, []byte("[DONE]")) {
		return
	}
	p.event(payload)
}

// streamEvent is a superset of the chunk shapes we care about.
type streamEvent struct {
	Type     string `json:"type"`
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Usage    *usage `json:"usage"`
	Choices  []struct {
		Delta struct {
			Content          *string         `json:"content"`
			ReasoningContent *string         `json:"reasoning_content"`
			Reasoning        *string         `json:"reasoning"`
			ToolCalls        json.RawMessage `json:"tool_calls"`
		} `json:"delta"`
		Text         *string `json:"text"` // legacy /v1/completions
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`

	// Anthropic: message_start carries model + input usage.
	Message *struct {
		Model string `json:"model"`
		Usage *usage `json:"usage"`
	} `json:"message"`
	// Anthropic sends an object here, the Responses API sends a string.
	Delta json.RawMessage `json:"delta"`

	// Responses API: response.completed carries model + usage.
	Response *struct {
		Model string `json:"model"`
		Usage *usage `json:"usage"`
	} `json:"response"`
}

type anthropicDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Thinking    string `json:"thinking"`
	PartialJSON string `json:"partial_json"`
	StopReason  string `json:"stop_reason"`
}

func (p *sseParser) event(payload []byte) {
	var ev streamEvent
	if err := json.Unmarshal(payload, &ev); err != nil {
		return
	}
	r := p.res
	r.events++
	now := p.now()

	if ev.Model != "" {
		r.model = ev.Model
	}
	if ev.Provider != "" {
		r.provider = ev.Provider
	}
	r.addUsage(ev.Usage)

	// OpenAI chat / legacy completions.
	for _, c := range ev.Choices {
		d := c.Delta
		switch {
		case d.Content != nil && *d.Content != "":
			r.markToken(now)
			r.appendText(*d.Content)
		case d.ReasoningContent != nil && *d.ReasoningContent != "":
			r.markToken(now)
		case d.Reasoning != nil && *d.Reasoning != "":
			r.markToken(now)
		case len(d.ToolCalls) > 0 && string(d.ToolCalls) != "null":
			r.markToken(now)
		case c.Text != nil && *c.Text != "":
			r.markToken(now)
			r.appendText(*c.Text)
		}
		if c.FinishReason != nil {
			r.addFinish(*c.FinishReason)
		}
	}

	switch {
	// Anthropic.
	case ev.Type == "message_start" && ev.Message != nil:
		if ev.Message.Model != "" {
			r.model = ev.Message.Model
		}
		r.addUsage(ev.Message.Usage)
	case ev.Type == "content_block_delta":
		var d anthropicDelta
		if json.Unmarshal(ev.Delta, &d) == nil {
			if d.Text != "" || d.Thinking != "" || d.PartialJSON != "" {
				r.markToken(now)
			}
			r.appendText(d.Text)
		}
	case ev.Type == "message_delta":
		var d anthropicDelta
		if json.Unmarshal(ev.Delta, &d) == nil {
			r.addFinish(d.StopReason)
		}

	// Responses API: any "*.delta" event with a string delta is model output.
	case strings.HasPrefix(ev.Type, "response.") && strings.HasSuffix(ev.Type, ".delta"):
		var s string
		if json.Unmarshal(ev.Delta, &s) == nil && s != "" {
			r.markToken(now)
			if ev.Type == "response.output_text.delta" {
				r.appendText(s)
			}
		}
	case ev.Type == "response.completed" || ev.Type == "response.incomplete":
		if ev.Response != nil {
			if ev.Response.Model != "" {
				r.model = ev.Response.Model
			}
			r.addUsage(ev.Response.Usage)
		}
		r.addFinish(strings.TrimPrefix(ev.Type, "response."))
	}
}

// ---- non-streaming JSON ----

type fullResponse struct {
	Model    string `json:"model"`
	Provider string `json:"provider"`
	Usage    *usage `json:"usage"`
	Choices  []struct {
		Message *struct {
			Content json.RawMessage `json:"content"`
		} `json:"message"`
		Text         string `json:"text"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	// Anthropic.
	Content    []contentPart `json:"content"`
	StopReason string        `json:"stop_reason"`
	// Responses API.
	Output []struct {
		Content []contentPart `json:"content"`
	} `json:"output"`
	Status string `json:"status"`
}

type contentPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

func parseFull(body []byte, r *result) {
	var fr fullResponse
	if err := json.Unmarshal(body, &fr); err != nil {
		return
	}
	r.model = fr.Model
	r.provider = fr.Provider
	r.addUsage(fr.Usage)
	for _, c := range fr.Choices {
		if c.Message != nil && len(c.Message.Content) > 0 {
			var s string
			if json.Unmarshal(c.Message.Content, &s) == nil {
				r.appendText(s)
			} else {
				var parts []contentPart
				if json.Unmarshal(c.Message.Content, &parts) == nil {
					for _, p := range parts {
						r.appendText(p.Text)
					}
				}
			}
		}
		r.appendText(c.Text)
		r.addFinish(c.FinishReason)
	}
	for _, p := range fr.Content {
		r.appendText(p.Text)
	}
	r.addFinish(fr.StopReason)
	for _, o := range fr.Output {
		for _, p := range o.Content {
			r.appendText(p.Text)
		}
	}
	if len(fr.Output) > 0 {
		r.addFinish(fr.Status)
	}
}

// ---- request side ----

// reqInfo is what we extract from the client request body.
type reqInfo struct {
	Model  string
	Stream bool
	User   string
	Params map[string]any
	Input  json.RawMessage // messages / input / prompt
	System json.RawMessage // Anthropic top-level system prompt
}

var paramKeys = []string{
	"temperature", "top_p", "top_k", "max_tokens", "max_completion_tokens",
	"max_output_tokens", "frequency_penalty", "presence_penalty", "seed",
	"reasoning_effort", "stop", "n",
}

// parseRequest decodes the body with UseNumber so we can re-encode it without
// mangling large integers.
func parseRequest(body []byte) (map[string]any, reqInfo, bool) {
	var m map[string]any
	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	if err := dec.Decode(&m); err != nil || m == nil {
		return nil, reqInfo{}, false
	}
	info := reqInfo{Params: map[string]any{}}
	info.Model, _ = m["model"].(string)
	info.Stream, _ = m["stream"].(bool)
	if u, ok := m["user"].(string); ok {
		info.User = u
	} else if md, ok := m["metadata"].(map[string]any); ok {
		info.User, _ = md["user_id"].(string)
	}
	for _, k := range paramKeys {
		if v, ok := m[k]; ok {
			info.Params[k] = v
		}
	}
	for _, k := range []string{"messages", "input", "prompt"} {
		if v, ok := m[k]; ok {
			info.Input, _ = json.Marshal(v)
			break
		}
	}
	if v, ok := m["system"]; ok {
		info.System, _ = json.Marshal(v)
	}
	return m, info, true
}

// injectIncludeUsage sets stream_options.include_usage=true so OpenAI-style
// providers report token usage in the final stream chunk. Returns true if the
// body was modified.
func injectIncludeUsage(m map[string]any) bool {
	so, _ := m["stream_options"].(map[string]any)
	if so == nil {
		so = map[string]any{}
	}
	if v, ok := so["include_usage"].(bool); ok && v {
		return false
	}
	so["include_usage"] = true
	m["stream_options"] = so
	return true
}
