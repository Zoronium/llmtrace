# Builds natively on whatever arch Coolify runs on (arm64 on Oracle Ampere, amd64 elsewhere).
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod ./
COPY . .
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w -X main.version=${VERSION}" -o /out/llmtrace ./cmd/llmtrace

FROM alpine:3.20
RUN adduser -D -u 10001 app
COPY --from=build /out/llmtrace /usr/local/bin/llmtrace
USER app
EXPOSE 8080
ENV LISTEN_ADDR=:8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD wget -qO- http://127.0.0.1:8080/healthz || exit 1
ENTRYPOINT ["llmtrace"]
