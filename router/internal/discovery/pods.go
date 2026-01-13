package discovery

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

// Pod represents a discovered rkllama pod
type Pod struct {
	Name     string
	IP       string
	NodeName string
	Ready    bool
	Models   []string // Models currently loaded on this pod
}

// Discovery watches rkllama pods and tracks their loaded models
type Discovery struct {
	client        *kubernetes.Clientset
	namespace     string
	labelSelector string
	httpClient    *http.Client

	mu       sync.RWMutex
	pods     map[string]*Pod          // pod name -> Pod
	modelMap map[string][]*Pod        // model name -> pods that have it
	rrIndex  map[string]int           // round-robin index per model
}

// TagsResponse represents the response from /api/tags
type TagsResponse struct {
	Models []ModelInfo `json:"models"`
}

// ModelInfo represents a model from /api/tags
type ModelInfo struct {
	Name       string `json:"name"`
	Model      string `json:"model"`
	ModifiedAt string `json:"modified_at"`
	Size       int64  `json:"size"`
}

// PsResponse represents the response from /api/ps (loaded models)
type PsResponse struct {
	Models []LoadedModelInfo `json:"models"`
}

// LoadedModelInfo represents a model from /api/ps
type LoadedModelInfo struct {
	Name     string `json:"name"`
	Model    string `json:"model"`
	Size     int64  `json:"size"`
	LoadedAt string `json:"loaded_at"`
}

// New creates a new Discovery instance
func New(ctx context.Context, namespace, labelSelector string) (*Discovery, error) {
	// Create in-cluster config
	config, err := rest.InClusterConfig()
	if err != nil {
		// Fall back to kubeconfig for local development
		slog.Warn("failed to get in-cluster config, using default", "error", err)
		config = &rest.Config{
			Host: "http://localhost:8001", // kubectl proxy
		}
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kubernetes client: %w", err)
	}

	return &Discovery{
		client:        client,
		namespace:     namespace,
		labelSelector: labelSelector,
		httpClient: &http.Client{
			Timeout: 5 * time.Second,
		},
		pods:     make(map[string]*Pod),
		modelMap: make(map[string][]*Pod),
		rrIndex:  make(map[string]int),
	}, nil
}

// Run starts the discovery loop
func (d *Discovery) Run(ctx context.Context) {
	ticker := time.NewTicker(10 * time.Second)
	defer ticker.Stop()

	// Initial discovery
	d.discover(ctx)

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.discover(ctx)
		}
	}
}

// discover fetches current pod state
func (d *Discovery) discover(ctx context.Context) {
	selector, err := labels.Parse(d.labelSelector)
	if err != nil {
		slog.Error("invalid label selector", "selector", d.labelSelector, "error", err)
		return
	}

	pods, err := d.client.CoreV1().Pods(d.namespace).List(ctx, metav1.ListOptions{
		LabelSelector: selector.String(),
	})
	if err != nil {
		slog.Error("failed to list pods", "error", err)
		return
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	// Reset state
	newPods := make(map[string]*Pod)
	newModelMap := make(map[string][]*Pod)

	for _, p := range pods.Items {
		if p.Status.Phase != corev1.PodRunning {
			continue
		}

		pod := &Pod{
			Name:     p.Name,
			IP:       p.Status.PodIP,
			NodeName: p.Spec.NodeName,
			Ready:    isPodReady(&p),
		}

		if pod.IP == "" || !pod.Ready {
			continue
		}

		// Fetch models from this pod
		models, err := d.fetchModels(ctx, pod.IP)
		if err != nil {
			slog.Debug("failed to fetch models from pod", "pod", pod.Name, "error", err)
			continue
		}
		pod.Models = models

		newPods[pod.Name] = pod

		// Update model map
		for _, model := range models {
			newModelMap[model] = append(newModelMap[model], pod)
		}
	}

	d.pods = newPods
	d.modelMap = newModelMap

	slog.Debug("discovery complete",
		"pods", len(d.pods),
		"models", len(d.modelMap),
	)
}

// fetchModels gets the list of loaded models from a pod's /api/ps endpoint
func (d *Discovery) fetchModels(ctx context.Context, podIP string) ([]string, error) {
	url := fmt.Sprintf("http://%s:8080/api/ps", podIP)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}

	resp, err := d.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status: %d", resp.StatusCode)
	}

	var ps PsResponse
	if err := json.NewDecoder(resp.Body).Decode(&ps); err != nil {
		return nil, err
	}

	models := make([]string, 0, len(ps.Models))
	for _, m := range ps.Models {
		models = append(models, m.Name)
	}

	return models, nil
}

// GetPodsForModel returns pods that have the specified model loaded
func (d *Discovery) GetPodsForModel(model string) []*Pod {
	d.mu.RLock()
	defer d.mu.RUnlock()

	return d.modelMap[model]
}

// GetNextPodForModel returns the next pod for round-robin load balancing
func (d *Discovery) GetNextPodForModel(model string) *Pod {
	d.mu.Lock()
	defer d.mu.Unlock()

	pods := d.modelMap[model]
	if len(pods) == 0 {
		return nil
	}

	idx := d.rrIndex[model] % len(pods)
	d.rrIndex[model] = idx + 1

	return pods[idx]
}

// GetAllPods returns all discovered pods
func (d *Discovery) GetAllPods() []*Pod {
	d.mu.RLock()
	defer d.mu.RUnlock()

	pods := make([]*Pod, 0, len(d.pods))
	for _, p := range d.pods {
		pods = append(pods, p)
	}
	return pods
}

// GetAllModels returns all known models across all pods
func (d *Discovery) GetAllModels() []string {
	d.mu.RLock()
	defer d.mu.RUnlock()

	models := make([]string, 0, len(d.modelMap))
	for model := range d.modelMap {
		models = append(models, model)
	}
	return models
}

// HasPods returns true if at least one pod is discovered
func (d *Discovery) HasPods() bool {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return len(d.pods) > 0
}

// GetPodByName returns a specific pod by name
func (d *Discovery) GetPodByName(name string) *Pod {
	d.mu.RLock()
	defer d.mu.RUnlock()
	return d.pods[name]
}

// isPodReady checks if a pod has the Ready condition
func isPodReady(pod *corev1.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == corev1.PodReady && cond.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
