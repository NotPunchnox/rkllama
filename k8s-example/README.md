# RKLlama Kubernetes Deployment

Kustomize manifests for deploying RKLlama on Kubernetes clusters with RK3588/RK3576 NPU nodes.

## Prerequisites

1. **Rockchip NPU Device Plugin**

   Install the RKNPU device plugin to expose NPU resources to Kubernetes:
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/rockchip-linux/rknpu-device-plugin/master/deploy/rknpu-device-plugin.yaml
   ```

   See: https://github.com/rockchip-linux/rknpu-device-plugin

2. **Node Taint (Optional)**

   If you want to dedicate NPU nodes for RKLlama workloads:
   ```bash
   kubectl taint nodes <node-name> npu=enabled:NoSchedule
   ```

3. **Ingress Controller** (for external access)

   The ingress manifest assumes nginx-ingress. Adjust annotations for your setup.

## Quick Start

```bash
# Create namespace and deploy
kubectl create namespace rkllama
kubectl apply -k ./k8s-example

# Watch deployment
kubectl -n rkllama get pods -w

# Check logs
kubectl -n rkllama logs -f deployment/rkllama
```

### Override Image Tag

Edit `kustomization.yml` to pin a specific version:

```yaml
images:
  - name: ghcr.io/rsjames-ttrpg/rkllama
    newTag: v0.1.0  # Change from 'latest'
```

Or update an existing deployment:

```bash
kubectl -n rkllama set image deployment/rkllama rkllama=ghcr.io/rsjames-ttrpg/rkllama:v0.1.0
```

## Manifests

| File | Description |
|------|-------------|
| `kustomization.yml` | Kustomize configuration, sets namespace |
| `deployment.yml` | Single-node deployment with NPU resource requests |
| `daemonset.yml` | Multi-node DaemonSet (one pod per NPU node) |
| `service.yml` | ClusterIP service on port 8080 |
| `ingress.yml` | Ingress with TLS and timeout settings |
| `pv.yml` | PersistentVolumeClaim for model storage |
| `router/` | Model-aware router for multi-node clusters |

## Router (Multi-Node Clusters)

For multi-node deployments, the router provides intelligent request routing based on which models are loaded on each pod.

### Router Features

- **Model-aware routing**: Routes requests to pods that have the requested model loaded
- **Round-robin load balancing**: Distributes requests across pods with the same model
- **Automatic model placement**: Configures which models should be loaded on which pods
- **Aggregated endpoints**: `/api/tags` shows all models across all pods

### Router Configuration

Edit `router/configmap.yml` to define model placement:

```yaml
models:
  # Replica-based: router picks nodes automatically
  qwen-7b:
    replicas: 2      # Load on 2 pods
    priority: 10     # Higher priority models placed first

  # Node-based: explicit placement
  llama-70b:
    nodes:
      - big-memory-node-1
      - big-memory-node-2

# Model aliases (Ollama-style names)
aliases:
  "qwen:7b": "qwen-7b"
  "qwen:latest": "qwen-7b"
```

### Disable Router

If you don't need the router (single-node deployment), comment it out in `kustomization.yml`:

```yaml
resources:
  - ./deployment.yml
  - ./service.yml
  # - ./router  # Disabled
```

## Configuration

### Resource Limits

The deployment requests:
- 1x NPU (`rockchip.com/npu: "1"`)
- 4-8 GB RAM
- 1-2 CPU cores

Adjust in `deployment.yml` based on your models:

```yaml
resources:
  limits:
    rockchip.com/npu: "1"
    memory: "16Gi"  # Increase for larger models
    cpu: "4"
  requests:
    memory: "8Gi"
    cpu: "2"
```

### Model Storage

Models are stored in a PersistentVolumeClaim mounted at `/opt/rkllama/models`.

To use a specific StorageClass:
```yaml
# pv.yml
spec:
  storageClassName: local-path  # or your preferred class
  resources:
    requests:
      storage: 50Gi  # Increase for multiple models
```

### Ingress

The default ingress is configured for:
- Host: `rkllama.home`
- TLS via cert-manager
- 300s timeouts (for slow model loading)

Customize for your domain:
```yaml
# ingress.yml
spec:
  rules:
    - host: llm.yourdomain.com
```

## Privileged Mode

The deployment runs in privileged mode with `/sys` mounted. This is required for the `fix_freq` script which optimizes NPU/CPU/GPU frequencies for maximum inference performance.

If you cannot run privileged containers, remove:
```yaml
securityContext:
  privileged: true
volumeMounts:
  - name: sys
    mountPath: /sys
```

Note: Performance may be reduced without frequency optimization.

## Testing

```bash
# Port forward for local testing
kubectl -n rkllama port-forward svc/rkllama 8080:8080

# List models
curl http://localhost:8080/api/tags

# Pull a model
curl -X POST http://localhost:8080/api/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "owner/repo/model.rkllm", "model_name": "qwen"}'

# Generate text
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen", "prompt": "Hello!"}'
```

## Troubleshooting

### Pod stuck in Pending

Check if NPU device plugin is running:
```bash
kubectl get pods -n kube-system | grep rknpu
kubectl describe node <node> | grep rockchip
```

### Model loading timeout

Increase ingress timeouts or use port-forward for initial model pulls:
```yaml
nginx.ingress.kubernetes.io/proxy-read-timeout: "600"
```

### Permission denied on /sys

Ensure privileged mode is enabled and your cluster allows it:
```bash
kubectl auth can-i create pods --as=system:serviceaccount:rkllama:default
```

## Integration with Ollama Clients

RKLlama is Ollama-compatible. Point any Ollama client to your ingress:

```bash
export OLLAMA_HOST=https://rkllama.home
ollama list
ollama run qwen "Hello!"
```
