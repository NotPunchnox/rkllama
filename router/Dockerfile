# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy go mod files
COPY go.mod go.sum* ./
RUN go mod download

# Copy source
COPY . .

# Build
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /router ./cmd/router

# Runtime stage
FROM alpine:3.20

RUN apk --no-cache add ca-certificates

COPY --from=builder /router /router

EXPOSE 8080

ENTRYPOINT ["/router"]
