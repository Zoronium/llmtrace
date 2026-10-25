// Package otlp is a tiny, dependency-free OTLP/HTTP JSON trace exporter.
//
// It implements just enough of the OTLP spec to ship spans to any OTLP/HTTP
// collector that accepts JSON (Langfuse, OpenTelemetry Collector, Jaeger, etc.).
// Spans are queued in a bounded channel and flushed in batches; when the queue
// is full new spans are dropped rather than blocking the request path.
package otlp

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// SpanKind values from the OTLP proto.
const (
	KindInternal = 1
	KindServer   = 2
	KindClient   = 3
)

// Status codes from the OTLP proto.
const (
	StatusUnset = 0
	StatusOK    = 1
	StatusError = 2
)

// TraceID and SpanID are raw identifiers, hex-encoded on the wire.
type (
	TraceID [16]byte
	SpanID  [8]byte
)

func (t TraceID) String() string { return hex.EncodeToString(t[:]) }
func (s SpanID) String() string  { return hex.EncodeToString(s[:]) }
func (t TraceID) IsZero() bool   { return t == TraceID{} }
func (s SpanID) IsZero() bool    { return s == SpanID{} }

// NewTraceID returns a random trace id.
func NewTraceID() TraceID {
	var t TraceID
	_, _ = rand.Read(t[:])
	return t
}

// NewSpanID returns a random span id.
func NewSpanID() SpanID {
	var s SpanID
	_, _ = rand.Read(s[:])
	return s
}

// ParseTraceparent parses a W3C traceparent header ("00-<trace>-<span>-<flags>").
func ParseTraceparent(h string) (TraceID, SpanID, bool) {
	var t TraceID
	var s SpanID
	parts := strings.Split(strings.TrimSpace(h), "-")
	if len(parts) != 4 || len(parts[1]) != 32 || len(parts[2]) != 16 {
		return t, s, false
	}
	tb, err1 := hex.DecodeString(parts[1])
	sb, err2 := hex.DecodeString(parts[2])
	if err1 != nil || err2 != nil {
		return t, s, false
	}
	copy(t[:], tb)
	copy(s[:], sb)
	if t.IsZero() || s.IsZero() {
		return t, s, false
	}
	return t, s, true
}

// Traceparent formats a sampled W3C traceparent header.
func Traceparent(t TraceID, s SpanID) string {
	return "00-" + t.String() + "-" + s.String() + "-01"
}

// Attr is a single span attribute.
type Attr struct {
	Key   string
	Value any // string, int, int64, float64, bool, []string
}

// String, Int, Float and Bool are small attribute constructors.
func String(k, v string) Attr           { return Attr{k, v} }
func Int(k string, v int64) Attr        { return Attr{k, v} }
func Float(k string, v float64) Attr    { return Attr{k, v} }
func Bool(k string, v bool) Attr        { return Attr{k, v} }
func Strings(k string, v []string) Attr { return Attr{k, v} }

// Span is a finished span ready for export.
type Span struct {
	TraceID       TraceID
	SpanID        SpanID
	ParentSpanID  SpanID
	Name          string
	Kind          int
	Start, End    time.Time
	Attrs         []Attr
	StatusCode    int
	StatusMessage string
}

// Exporter receives finished spans.
type Exporter interface {
	Export(Span)
	Shutdown(context.Context) error
}

// Noop discards spans. Used when no endpoint is configured.
type Noop struct{}

func (Noop) Export(Span)                    {}
func (Noop) Shutdown(context.Context) error { return nil }

// HTTPExporter batches spans and POSTs them as OTLP JSON.
type HTTPExporter struct {
	endpoint string
	headers  map[string]string
	resource []Attr
	client   *http.Client
	log      *slog.Logger

	queue   chan Span
	dropped atomic.Int64
	stop    chan struct{}
	done    chan struct{}
	once    sync.Once

	batchSize int
	interval  time.Duration
}

// Options configures an HTTPExporter.
type Options struct {
	Endpoint    string            // full traces URL, e.g. https://host/api/public/otel/v1/traces
	Headers     map[string]string // e.g. Authorization
	ServiceName string
	Version     string
	QueueSize   int
	BatchSize   int
	Interval    time.Duration
	Logger      *slog.Logger
}

// NewHTTP starts a batching exporter.
func NewHTTP(o Options) *HTTPExporter {
	if o.QueueSize <= 0 {
		o.QueueSize = 4096
	}
	if o.BatchSize <= 0 {
		o.BatchSize = 256
	}
	if o.Interval <= 0 {
		o.Interval = 2 * time.Second
	}
	if o.Logger == nil {
		o.Logger = slog.Default()
	}
	e := &HTTPExporter{
		endpoint:  o.Endpoint,
		headers:   o.Headers,
		client:    &http.Client{Timeout: 15 * time.Second},
		log:       o.Logger,
		queue:     make(chan Span, o.QueueSize),
		stop:      make(chan struct{}),
		done:      make(chan struct{}),
		batchSize: o.BatchSize,
		interval:  o.Interval,
		resource: []Attr{
			String("service.name", o.ServiceName),
			String("service.version", o.Version),
			String("telemetry.sdk.name", "llmtrace-otlp"),
			String("telemetry.sdk.language", "go"),
		},
	}
	go e.run()
	return e
}

// FromEnv builds an exporter from the standard OTEL_EXPORTER_OTLP_* variables.
// It returns a Noop exporter when no endpoint is configured.
func FromEnv(service, version string, log *slog.Logger) Exporter {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
	if endpoint == "" {
		if base := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT"); base != "" {
			endpoint = strings.TrimRight(base, "/") + "/v1/traces"
		}
	}
	if endpoint == "" {
		log.Warn("no OTLP endpoint configured, spans will only be logged")
		return Noop{}
	}
	h := os.Getenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
	if h == "" {
		h = os.Getenv("OTEL_EXPORTER_OTLP_HEADERS")
	}
	if s := os.Getenv("OTEL_SERVICE_NAME"); s != "" {
		service = s
	}
	log.Info("OTLP exporter enabled", "endpoint", endpoint)
	return NewHTTP(Options{
		Endpoint:    endpoint,
		Headers:     ParseHeaders(h),
		ServiceName: service,
		Version:     version,
		Logger:      log,
	})
}

// ParseHeaders parses "k1=v1,k2=v2" with URL-encoded values, splitting on the
// first '=' only so base64 padding survives.
func ParseHeaders(s string) map[string]string {
	out := map[string]string{}
	for _, pair := range strings.Split(s, ",") {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		k, v, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		if dv, err := url.PathUnescape(strings.TrimSpace(v)); err == nil {
			v = dv
		}
		out[strings.TrimSpace(k)] = strings.TrimSpace(v)
	}
	return out
}

// Export enqueues a span without blocking. Spans are dropped if the queue is full.
func (e *HTTPExporter) Export(s Span) {
	select {
	case e.queue <- s:
	default:
		if n := e.dropped.Add(1); n == 1 || n%100 == 0 {
			e.log.Warn("span queue full, dropping spans", "dropped_total", n)
		}
	}
}

// Shutdown flushes queued spans and stops the exporter.
func (e *HTTPExporter) Shutdown(ctx context.Context) error {
	e.once.Do(func() { close(e.stop) })
	select {
	case <-e.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (e *HTTPExporter) run() {
	defer close(e.done)
	t := time.NewTicker(e.interval)
	defer t.Stop()
	batch := make([]Span, 0, e.batchSize)
	flush := func() {
		if len(batch) == 0 {
			return
		}
		if err := e.send(batch); err != nil {
			e.log.Error("OTLP export failed", "spans", len(batch), "err", err)
		}
		batch = batch[:0]
	}
	for {
		select {
		case s := <-e.queue:
			batch = append(batch, s)
			if len(batch) >= e.batchSize {
				flush()
			}
		case <-t.C:
			flush()
		case <-e.stop:
			for {
				select {
				case s := <-e.queue:
					batch = append(batch, s)
					if len(batch) >= e.batchSize {
						flush()
					}
				default:
					flush()
					return
				}
			}
		}
	}
}

func (e *HTTPExporter) send(spans []Span) error {
	body, err := Encode(e.resource, spans)
	if err != nil {
		return err
	}
	var lastErr error
	for attempt := 0; attempt < 3; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(attempt) * 500 * time.Millisecond)
		}
		req, err := http.NewRequest(http.MethodPost, e.endpoint, bytes.NewReader(body))
		if err != nil {
			return err
		}
		req.Header.Set("Content-Type", "application/json")
		for k, v := range e.headers {
			req.Header.Set(k, v)
		}
		resp, err := e.client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		resp.Body.Close()
		if resp.StatusCode/100 == 2 {
			return nil
		}
		lastErr = fmt.Errorf("status %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))
		if resp.StatusCode != http.StatusTooManyRequests && resp.StatusCode < 500 {
			return lastErr // not retryable
		}
	}
	return lastErr
}

// ---- OTLP JSON encoding ----

type anyValue map[string]any

func encodeValue(v any) (anyValue, error) {
	switch x := v.(type) {
	case string:
		return anyValue{"stringValue": x}, nil
	case bool:
		return anyValue{"boolValue": x}, nil
	case int:
		return anyValue{"intValue": strconv.FormatInt(int64(x), 10)}, nil
	case int64:
		return anyValue{"intValue": strconv.FormatInt(x, 10)}, nil
	case float64:
		return anyValue{"doubleValue": x}, nil
	case []string:
		vals := make([]anyValue, len(x))
		for i, s := range x {
			vals[i] = anyValue{"stringValue": s}
		}
		return anyValue{"arrayValue": map[string]any{"values": vals}}, nil
	default:
		return nil, fmt.Errorf("unsupported attribute type %T", v)
	}
}

func encodeAttrs(attrs []Attr) []map[string]any {
	out := make([]map[string]any, 0, len(attrs))
	for _, a := range attrs {
		v, err := encodeValue(a.Value)
		if err != nil || a.Key == "" {
			continue
		}
		out = append(out, map[string]any{"key": a.Key, "value": v})
	}
	return out
}

// Encode renders spans as an OTLP ExportTraceServiceRequest in JSON.
func Encode(resource []Attr, spans []Span) ([]byte, error) {
	if len(spans) == 0 {
		return nil, errors.New("no spans")
	}
	js := make([]map[string]any, 0, len(spans))
	for _, s := range spans {
		m := map[string]any{
			"traceId":           s.TraceID.String(),
			"spanId":            s.SpanID.String(),
			"name":              s.Name,
			"kind":              s.Kind,
			"startTimeUnixNano": strconv.FormatInt(s.Start.UnixNano(), 10),
			"endTimeUnixNano":   strconv.FormatInt(s.End.UnixNano(), 10),
			"attributes":        encodeAttrs(s.Attrs),
			"status":            map[string]any{"code": s.StatusCode, "message": s.StatusMessage},
		}
		if !s.ParentSpanID.IsZero() {
			m["parentSpanId"] = s.ParentSpanID.String()
		}
		js = append(js, m)
	}
	req := map[string]any{
		"resourceSpans": []any{map[string]any{
			"resource": map[string]any{"attributes": encodeAttrs(resource)},
			"scopeSpans": []any{map[string]any{
				"scope": map[string]any{"name": "llmtrace"},
				"spans": js,
			}},
		}},
	}
	return json.Marshal(req)
}
