package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"

	"github.com/rsjames-ttrpg/rkllama/router/internal/discovery"
)

// Handler handles HTTP requests and proxies them to appropriate pods
type Handler struct {
	discovery *discovery.Discovery
}

// NewHandler creates a new proxy handler
func NewHandler(disc *discovery.Discovery) *Handler {
	return &Handler{
		discovery: disc,
	}
}

// Health handles liveness probe
func (h *Handler) Health(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte(`{"status":"ok"}`))
}

// Ready handles readiness probe
func (h *Handler) Ready(w http.ResponseWriter, r *http.Request) {
	if !h.discovery.HasPods() {
		w.WriteHeader(http.StatusServiceUnavailable)
		w.Write([]byte(`{"status":"no pods discovered"}`))
		return
	}
	w.WriteHeader(http.StatusOK)
	json.NewEncoder(w).Encode(map[string]any{
		"status": "ok",
		"pods":   len(h.discovery.GetAllPods()),
		"models": h.discovery.GetAllModels(),
	})
}

// Proxy handles all other requests
func (h *Handler) Proxy(w http.ResponseWriter, r *http.Request) {
	// Special handling for aggregate endpoints
	if r.URL.Path == "/api/tags" && r.Method == http.MethodGet {
		h.aggregateTags(w, r)
		return
	}
	if r.URL.Path == "/api/ps" && r.Method == http.MethodGet {
		h.aggregatePS(w, r)
		return
	}
	// Router status endpoints
	if r.URL.Path == "/router/status" && r.Method == http.MethodGet {
		h.routerStatus(w, r)
		return
	}
	if r.URL.Path == "/router/pods" && r.Method == http.MethodGet {
		h.routerPods(w, r)
		return
	}
	if r.URL.Path == "/router/models" && r.Method == http.MethodGet {
		h.routerModels(w, r)
		return
	}

	// Extract model from request
	model, body, err := h.extractModel(r)
	if err != nil {
		slog.Debug("failed to extract model", "error", err)
		// Fall back to any pod if model extraction fails
		h.proxyToAnyPod(w, r, body)
		return
	}

	if model == "" {
		// No model specified, proxy to any pod
		h.proxyToAnyPod(w, r, body)
		return
	}

	// Find a pod with the model loaded
	pod := h.discovery.GetNextPodForModel(model)
	if pod == nil {
		slog.Warn("no pod found for model", "model", model)
		http.Error(w, fmt.Sprintf("model %q not loaded on any pod", model), http.StatusServiceUnavailable)
		return
	}

	slog.Debug("routing request", "model", model, "pod", pod.Name, "path", r.URL.Path)
	h.proxyToPod(w, r, pod, body)
}

// extractModel reads the request body and extracts the model field
func (h *Handler) extractModel(r *http.Request) (string, []byte, error) {
	if r.Body == nil {
		return "", nil, nil
	}

	body, err := io.ReadAll(r.Body)
	if err != nil {
		return "", nil, err
	}
	r.Body.Close()

	if len(body) == 0 {
		return "", body, nil
	}

	// Try to parse as JSON to extract model
	var req struct {
		Model string `json:"model"`
	}
	if err := json.Unmarshal(body, &req); err != nil {
		// Not JSON or no model field, that's ok
		return "", body, nil
	}

	return req.Model, body, nil
}

// proxyToAnyPod proxies to any available pod
func (h *Handler) proxyToAnyPod(w http.ResponseWriter, r *http.Request, body []byte) {
	pods := h.discovery.GetAllPods()
	if len(pods) == 0 {
		http.Error(w, "no pods available", http.StatusServiceUnavailable)
		return
	}

	// Simple round-robin across all pods
	pod := pods[0] // TODO: proper round-robin for non-model requests
	h.proxyToPod(w, r, pod, body)
}

// proxyToPod proxies the request to a specific pod
func (h *Handler) proxyToPod(w http.ResponseWriter, r *http.Request, pod *discovery.Pod, body []byte) {
	target := &url.URL{
		Scheme: "http",
		Host:   fmt.Sprintf("%s:8080", pod.IP),
	}

	proxy := &httputil.ReverseProxy{
		Director: func(req *http.Request) {
			req.URL.Scheme = target.Scheme
			req.URL.Host = target.Host
			req.Host = target.Host

			// Restore body if we read it
			if body != nil {
				req.Body = io.NopCloser(bytes.NewReader(body))
				req.ContentLength = int64(len(body))
			}
		},
		// Streaming support: don't buffer response
		FlushInterval: -1,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			slog.Error("proxy error", "pod", pod.Name, "error", err)
			http.Error(w, "proxy error: "+err.Error(), http.StatusBadGateway)
		},
	}

	proxy.ServeHTTP(w, r)
}

// aggregateTags aggregates /api/tags from all pods
func (h *Handler) aggregateTags(w http.ResponseWriter, _ *http.Request) {
	pods := h.discovery.GetAllPods()
	if len(pods) == 0 {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"models":[]}`))
		return
	}

	// Collect unique models across all pods
	seen := make(map[string]discovery.ModelInfo)
	for _, pod := range pods {
		for _, model := range pod.Models {
			if _, ok := seen[model]; !ok {
				seen[model] = discovery.ModelInfo{
					Name:  model,
					Model: model,
				}
			}
		}
	}

	models := make([]discovery.ModelInfo, 0, len(seen))
	for _, m := range seen {
		models = append(models, m)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"models": models,
	})
}

// aggregatePS aggregates /api/ps from all pods
func (h *Handler) aggregatePS(w http.ResponseWriter, _ *http.Request) {
	pods := h.discovery.GetAllPods()

	type runningModel struct {
		Name     string `json:"name"`
		Model    string `json:"model"`
		NodeName string `json:"node_name"`
		PodName  string `json:"pod_name"`
	}

	running := make([]runningModel, 0)
	for _, pod := range pods {
		for _, model := range pod.Models {
			running = append(running, runningModel{
				Name:     model,
				Model:    model,
				NodeName: pod.NodeName,
				PodName:  pod.Name,
			})
		}
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"models": running,
	})
}

// routerStatus returns detailed router status
func (h *Handler) routerStatus(w http.ResponseWriter, _ *http.Request) {
	pods := h.discovery.GetAllPods()

	type podStatus struct {
		Name     string   `json:"name"`
		IP       string   `json:"ip"`
		NodeName string   `json:"node_name"`
		Ready    bool     `json:"ready"`
		Models   []string `json:"models"`
	}

	podStatuses := make([]podStatus, 0, len(pods))
	totalModels := 0
	for _, pod := range pods {
		podStatuses = append(podStatuses, podStatus{
			Name:     pod.Name,
			IP:       pod.IP,
			NodeName: pod.NodeName,
			Ready:    pod.Ready,
			Models:   pod.Models,
		})
		totalModels += len(pod.Models)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status":        "ok",
		"pods":          podStatuses,
		"pod_count":     len(pods),
		"model_count":   len(h.discovery.GetAllModels()),
		"loaded_count":  totalModels,
		"unique_models": h.discovery.GetAllModels(),
	})
}

// routerPods returns detailed information about all pods
func (h *Handler) routerPods(w http.ResponseWriter, _ *http.Request) {
	pods := h.discovery.GetAllPods()

	type podInfo struct {
		Name       string   `json:"name"`
		IP         string   `json:"ip"`
		NodeName   string   `json:"node_name"`
		Ready      bool     `json:"ready"`
		Models     []string `json:"models"`
		ModelCount int      `json:"model_count"`
	}

	podInfos := make([]podInfo, 0, len(pods))
	for _, pod := range pods {
		podInfos = append(podInfos, podInfo{
			Name:       pod.Name,
			IP:         pod.IP,
			NodeName:   pod.NodeName,
			Ready:      pod.Ready,
			Models:     pod.Models,
			ModelCount: len(pod.Models),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"pods": podInfos,
	})
}

// routerModels returns models with their pod locations
func (h *Handler) routerModels(w http.ResponseWriter, _ *http.Request) {
	models := h.discovery.GetAllModels()

	type modelInfo struct {
		Name     string   `json:"name"`
		Pods     []string `json:"pods"`
		Nodes    []string `json:"nodes"`
		Replicas int      `json:"replicas"`
	}

	modelInfos := make([]modelInfo, 0, len(models))
	for _, model := range models {
		pods := h.discovery.GetPodsForModel(model)
		podNames := make([]string, 0, len(pods))
		nodeNames := make([]string, 0, len(pods))
		nodeSet := make(map[string]bool)

		for _, pod := range pods {
			podNames = append(podNames, pod.Name)
			if !nodeSet[pod.NodeName] {
				nodeSet[pod.NodeName] = true
				nodeNames = append(nodeNames, pod.NodeName)
			}
		}

		modelInfos = append(modelInfos, modelInfo{
			Name:     model,
			Pods:     podNames,
			Nodes:    nodeNames,
			Replicas: len(pods),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"models": modelInfos,
	})
}
