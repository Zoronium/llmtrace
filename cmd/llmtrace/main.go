// Command llmtrace is a transparent tracing proxy for OpenAI/Anthropic-compatible
// LLM gateways. It measures TTFB, TTFT, throughput and token usage for every
// call and exports OTLP spans (Langfuse-compatible) without touching the stream.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/Zoronium/llmtrace/internal/otlp"
	"github.com/Zoronium/llmtrace/internal/proxy"
)

var version = "dev"

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	upstreamRaw := os.Getenv("UPSTREAM_URL")
	if upstreamRaw == "" {
		log.Error("UPSTREAM_URL is required, e.g. http://freellmapi:3001")
		os.Exit(1)
	}
	upstream, err := url.Parse(upstreamRaw)
	if err != nil || upstream.Scheme == "" || upstream.Host == "" {
		log.Error("invalid UPSTREAM_URL", "value", upstreamRaw)
		os.Exit(1)
	}

	cfg := proxy.Config{
		Upstream:           upstream,
		CaptureContent:     envBool("CAPTURE_CONTENT", false),
		MaxCaptureBytes:    envInt("MAX_CAPTURE_BYTES", 32<<10),
		MaxRequestBytes:    int64(envInt("MAX_REQUEST_BYTES", 32<<20)),
		InjectUsage:        envBool("INJECT_INCLUDE_USAGE", true),
		Environment:        os.Getenv("LLMTRACE_ENVIRONMENT"),
		ForwardTraceparent: envBool("FORWARD_TRACEPARENT", true),
	}

	exp := otlp.FromEnv("llmtrace", version, log)
	h := proxy.New(cfg, exp, log)

	addr := os.Getenv("LISTEN_ADDR")
	if addr == "" {
		addr = ":8080"
	}
	srv := &http.Server{
		Addr:              addr,
		Handler:           h,
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		// No WriteTimeout: streamed generations can run for minutes.
	}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Info("llmtrace listening",
			"addr", addr, "upstream", upstream.String(), "version", version,
			"capture_content", cfg.CaptureContent)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server error", "err", err)
			os.Exit(1)
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_ = srv.Shutdown(shutdownCtx)
	if err := exp.Shutdown(shutdownCtx); err != nil {
		log.Error("exporter flush failed", "err", err)
	}
}

func envBool(k string, def bool) bool {
	if v, err := strconv.ParseBool(os.Getenv(k)); err == nil {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v, err := strconv.Atoi(os.Getenv(k)); err == nil && v > 0 {
		return v
	}
	return def
}
