package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/rsjames-ttrpg/rkllama/router/internal/discovery"
	"github.com/rsjames-ttrpg/rkllama/router/internal/proxy"
	"github.com/rsjames-ttrpg/rkllama/router/internal/reconciler"
)

func main() {
	var (
		port          = flag.String("port", "8080", "Port to listen on")
		namespace     = flag.String("namespace", "default", "Kubernetes namespace to watch")
		labelSelector = flag.String("selector", "app=rkllama", "Label selector for rkllama pods")
		syncInterval  = flag.Duration("sync-interval", 30*time.Second, "Interval between reconciliation runs")
		configMap     = flag.String("config", "rkllama-model-config", "ConfigMap name for model assignments")
	)
	flag.Parse()

	// Setup structured logging
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	}))
	slog.SetDefault(logger)

	slog.Info("starting rkllama-router",
		"port", *port,
		"namespace", *namespace,
		"selector", *labelSelector,
	)

	// Create context that cancels on interrupt
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		slog.Info("received shutdown signal", "signal", sig)
		cancel()
	}()

	// Initialize pod discovery
	disc, err := discovery.New(ctx, *namespace, *labelSelector)
	if err != nil {
		slog.Error("failed to initialize discovery", "error", err)
		os.Exit(1)
	}

	// Start discovery loop
	go disc.Run(ctx)

	// Initialize reconciler
	recon, err := reconciler.New(ctx, *namespace, *configMap, disc)
	if err != nil {
		slog.Error("failed to initialize reconciler", "error", err)
		os.Exit(1)
	}

	// Start reconciler loop
	go recon.Run(ctx, *syncInterval)

	// Initialize proxy handler
	handler := proxy.NewHandler(disc)

	// Setup HTTP server
	mux := http.NewServeMux()
	mux.HandleFunc("/health", handler.Health)
	mux.HandleFunc("/health/ready", handler.Ready)
	mux.HandleFunc("/", handler.Proxy)

	server := &http.Server{
		Addr:         ":" + *port,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 0, // Disable for streaming
		IdleTimeout:  120 * time.Second,
	}

	// Start server
	go func() {
		slog.Info("starting HTTP server", "addr", server.Addr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			cancel()
		}
	}()

	// Wait for shutdown
	<-ctx.Done()

	// Graceful shutdown
	shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownCancel()

	slog.Info("shutting down server")
	if err := server.Shutdown(shutdownCtx); err != nil {
		slog.Error("shutdown error", "error", err)
	}

	slog.Info("router stopped")
}
