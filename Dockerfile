# Build stage
FROM golang:1.24-alpine AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY main.go ./
COPY cmd/ ./cmd/
COPY game/ ./game/
COPY ui/ ./ui/
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/wriggle .

# Runtime stage
FROM alpine:3.21
RUN apk add --no-cache ca-certificates
WORKDIR /app
COPY --from=builder /out/wriggle /app/wriggle
COPY models/wriggle-v4-personalities.best.json /app/models/wriggle-v4-personalities.best.json
COPY models/wriggle-v3.best.json /app/models/wriggle-v3.best.json
RUN mkdir -p /app/.ssh
VOLUME ["/app/.ssh"]
EXPOSE 23234
ENTRYPOINT ["/app/wriggle"]
CMD ["-port", "23234", "-bot-policy", "models/wriggle-v4-personalities.best.json"]
