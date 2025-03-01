# Quantization Format Mapping

This document explains how RKLLAMA maps its native quantization formats to Ollama-compatible formats for API compatibility.

## RKLLM to Ollama Quantization Mapping

| RKLLM Format | Ollama Format | Description |
|--------------|---------------|-------------|
| w4a16        | Q4_0          | 4-bit weights, 16-bit activations |
| w4a16_g32    | Q4_K_M        | 4-bit weights with groupsize 32, 16-bit activations |
| w4a16_g64    | Q4_K_M        | 4-bit weights with groupsize 64, 16-bit activations |
| w4a16_g128   | Q4_K_M        | 4-bit weights with groupsize 128, 16-bit activations |
| w8a8         | Q8_0          | 8-bit weights, 8-bit activations |
| w8a8_g128    | Q8_K_M        | 8-bit weights with groupsize 128, 8-bit activations |
| w8a8_g256    | Q8_K_M        | 8-bit weights with groupsize 256, 8-bit activations |
| w8a8_g512    | Q8_K_M        | 8-bit weights with groupsize 512, 8-bit activations |

## Understanding the Format

### RKLLM Format
In RKLLM, the quantization format follows the pattern `w{weight_bits}a{activation_bits}[_g{group_size}]`:
- `w` represents the weight bits (4-bit or 8-bit)
- `a` represents the activation bits (8-bit or 16-bit)
- `g` represents the group size (when present)

### Ollama Format
In Ollama, the quantization format follows patterns like `Q4_0`, `Q8_0`, `Q4_K_M`, etc:
- `Q4` represents 4-bit quantization
- `Q8` represents 8-bit quantization
- `K` represents special handling for keys and values
- `M` represents a multi-group approach that's similar to RKLLM's group sizes

## Hybrid Quantization

Some RKLLM models use hybrid quantization, indicated by "hybrid-ratio" in their names. This means a portion of the layers use one quantization type, while the rest use another type.

In the Ollama API, these are represented with a "-hybrid" suffix (e.g., `Q4_0-hybrid`).

## Parameter Size

The parameter size is extracted from the model name when available (like "3B", "7B", etc.) and represented consistently in the API responses.
