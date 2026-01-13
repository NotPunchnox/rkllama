# RKLlama Router

Model-aware request router for RKLlama DaemonSet deployments. Routes requests to pods that have the requested model loaded.

## Features

- **Model-aware routing**: Extracts model name from requests and routes to pods with that model loaded
- **Round-robin load balancing**: Distributes requests across pods with the same model
- **Pod discovery**: Watches Kubernetes for rkllama pods and polls `/api/tags` for loaded models
- **ConfigMap-driven reconciliation**: Automatically loads/unloads models to match desired state
- **Aggregated endpoints**: `/api/tags` and `/api/ps` show models across all pods
- **Streaming support**: Transparent proxying of SSE streaming responses

## Architecture

```
Client → Router → Pod (with model)

The router:
1. Receives request with {"model": "qwen-7b", ...}
2. Looks up pods that have qwen-7b loaded
3. Round-robin selects a pod
4. Proxies request to selected pod
5. Streams response back to client
```

## Quick Start

### Deploy to Kubernetes

```bash
# Deploy router alongside your existing rkllama DaemonSet
kubectl apply -k router/deploy/

# Verify
kubectl get pods -l app=rkllama-router
kubectl logs -f deployment/rkllama-router
```

### Configure Model Assignments

Edit the ConfigMap to define your desired model distribution:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rkllama-model-config
data:
  models.yaml: |
    models:
      # Replica-based: "I want 2 copies, don't care where"
      qwen-7b:
        replicas: 2
        priority: 10  # Higher priority models placed first

      qwen-1.5b:
        replicas: 4
        priority: 5

      # Node-based: "I want this on these specific nodes"
      llama-70b:
        nodes:
          - big-memory-node-1
          - big-memory-node-2
```

**Placement modes:**
- `replicas: N` - Router picks N nodes automatically, balancing load
- `nodes: [...]` - Explicit placement on specific nodes
- `priority: N` - Higher priority models get placed first (default: 0)

The router will automatically load models to match the config. For replica-based placement, it prefers nodes with fewer models already loaded.

### Use the Router

Point your clients at the router service instead of individual pods:

```bash
# List all models across all pods
curl http://rkllama-router:8080/api/tags

# Chat with a model (routed to appropriate pod)
curl http://rkllama-router:8080/api/chat \
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello!"}]}'

# OpenAI-compatible endpoint
curl http://rkllama-router:8080/v1/chat/completions \
  -d '{"model": "qwen-7b", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### Monitor the Cluster

```bash
# Get router status overview
curl http://rkllama-router:8080/router/status

# List all pods and their loaded models
curl http://rkllama-router:8080/router/pods

# List models and which nodes have them
curl http://rkllama-router:8080/router/models

# See running models with pod/node info (Ollama-compatible)
curl http://rkllama-router:8080/api/ps
```

### Emergency Recovery

If models get stuck loading or workers hang:

```bash
# Force kill all workers on all nodes
curl -X POST http://rkllama-router:8080/router/force_unload_all
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | `8080` | Port to listen on |
| `--namespace` | `default` | Kubernetes namespace to watch |
| `--selector` | `app=rkllama` | Label selector for rkllama pods |
| `--config` | `rkllama-model-config` | ConfigMap name for model assignments |
| `--sync-interval` | `30s` | How often to reconcile model state |

## Endpoints

### Health & Probes

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe (returns available pods/models) |

### Aggregated API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tags` | GET | Aggregated list of all models across all pods |
| `/api/ps` | GET | Running models with pod/node info |

### Router Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/router/status` | GET | Detailed router status with pod and model counts |
| `/router/pods` | GET | List all discovered pods with their loaded models |
| `/router/models` | GET | List all models with their pod/node locations |
| `/router/force_unload_all` | POST | Force kill all workers on all nodes (emergency recovery) |

### Proxied Requests

All other requests are proxied to the appropriate pod based on the `model` field in the request body.

#### Force Unload All

Use this endpoint when workers are stuck and normal unloading doesn't work:

```bash
curl -X POST http://rkllama-router:8080/router/force_unload_all
```

Response:
```json
{
  "message": "Force unload completed on 3/3 pods",
  "total_pods": 3,
  "success_count": 3,
  "results": [
    {
      "pod_name": "rkllama-abc12",
      "node_name": "node-1",
      "success": true,
      "killed_tracked": ["qwen-7b"],
      "killed_orphaned": []
    }
  ]
}
```

## Local Development

```bash
cd router

# Run with kubectl proxy for k8s API access
kubectl proxy &
go run ./cmd/router --namespace=rkllama

# Or build and run
go build -o router ./cmd/router
./router --namespace=rkllama
```

## Building

```bash
# Build binary
go build -o router ./cmd/router

# Build Docker image
docker build -t rkllama-router:latest .
```

## How It Works

1. **Discovery**: Every 10s, the router lists pods with `app=rkllama` label and queries each pod's `/api/tags` to see what models are loaded.

2. **Routing**: When a request comes in, the router:
   - Parses the request body to extract the `model` field
   - Looks up which pods have that model
   - Selects one using round-robin
   - Proxies the request using `httputil.ReverseProxy`

3. **Reconciliation**: Every 30s (configurable), the router:
   - Reads the ConfigMap for desired model assignments
   - Compares with actual state from discovery
   - Calls `/api/load` on pods that should have a model but don't
   - Calls `/api/unload` on pods that shouldn't have a model
