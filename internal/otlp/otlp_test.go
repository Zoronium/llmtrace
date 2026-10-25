package otlp

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"
)

func TestParseHeadersKeepsBase64Padding(t *testing.T) {
	h := ParseHeaders("Authorization=Basic%20cGstbGY6c2stbGY=,x-langfuse-ingestion-version=4")
	if h["Authorization"] != "Basic cGstbGY6c2stbGY=" {
		t.Errorf("auth = %q", h["Authorization"])
	}
	if h["x-langfuse-ingestion-version"] != "4" {
		t.Errorf("version = %q", h["x-langfuse-ingestion-version"])
	}
}

func TestExporterSendsOTLPJSON(t *testing.T) {
	var mu sync.Mutex
	var bodies [][]byte
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, b)
		auth = r.Header.Get("Authorization")
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	e := NewHTTP(Options{
		Endpoint:    srv.URL + "/v1/traces",
		Headers:     map[string]string{"Authorization": "Basic abc"},
		ServiceName: "test",
		Interval:    time.Hour, // force flush on shutdown
		Logger:      slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
	now := time.Now()
	e.Export(Span{
		TraceID: NewTraceID(), SpanID: NewSpanID(), Name: "chat m", Kind: KindClient,
		Start: now, End: now.Add(time.Second),
		Attrs:      []Attr{String("a", "b"), Int("n", 5), Float("f", 1.5), Bool("ok", true), Strings("l", []string{"x"})},
		StatusCode: StatusOK,
	})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := e.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}

	mu.Lock()
	defer mu.Unlock()
	if len(bodies) != 1 || auth != "Basic abc" {
		t.Fatalf("bodies=%d auth=%q", len(bodies), auth)
	}
	var req struct {
		ResourceSpans []struct {
			ScopeSpans []struct {
				Spans []struct {
					TraceID    string `json:"traceId"`
					SpanID     string `json:"spanId"`
					Start      string `json:"startTimeUnixNano"`
					Attributes []struct {
						Key   string         `json:"key"`
						Value map[string]any `json:"value"`
					} `json:"attributes"`
				} `json:"spans"`
			} `json:"scopeSpans"`
		} `json:"resourceSpans"`
	}
	if err := json.Unmarshal(bodies[0], &req); err != nil {
		t.Fatal(err)
	}
	sp := req.ResourceSpans[0].ScopeSpans[0].Spans[0]
	if len(sp.TraceID) != 32 || len(sp.SpanID) != 16 || sp.Start == "" {
		t.Errorf("bad ids/timestamps: %+v", sp)
	}
	vals := map[string]map[string]any{}
	for _, a := range sp.Attributes {
		vals[a.Key] = a.Value
	}
	if vals["n"]["intValue"] != "5" || vals["f"]["doubleValue"] != 1.5 || vals["ok"]["boolValue"] != true {
		t.Errorf("attribute encoding wrong: %v", vals)
	}
}

func TestParseTraceparent(t *testing.T) {
	tid, sid, ok := ParseTraceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")
	if !ok || tid.String() != "4bf92f3577b34da6a3ce929d0e0e4736" || sid.String() != "00f067aa0ba902b7" {
		t.Fatal("valid traceparent rejected")
	}
	if _, _, ok := ParseTraceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01"); ok {
		t.Error("all-zero trace id accepted")
	}
	if _, _, ok := ParseTraceparent("garbage"); ok {
		t.Error("garbage accepted")
	}
}
