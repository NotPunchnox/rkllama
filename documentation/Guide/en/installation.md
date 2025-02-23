# RKLLama Installation Guide

This guide provides step-by-step instructions to install **RKLLama**, a solution optimized for running AI and deep learning models on Rockchip devices with NPU support.

## Prerequisites

- **Hardware:** Device with Rockchip RK3588(S) or RK3576 processor
- **Operating System:** Ubuntu arm64
- **Internet Connection:** Required for downloading dependencies and models

## Installation Steps

1. **Clone the RKLLama Repository:**

```bash
   git clone https://github.com/notpunchnox/rkllama
   cd rkllama
```


2. **Run the Setup Script:**
```bash
   chmod +x setup.sh
   sudo ./setup.sh
```


   *Note:* To install without Miniconda, execute `sudo ./setup.sh --no-conda`.

## Post-Installation

- **Start the Server:**

```bash
  rkllama serve
```

- **Access the Client:**

```bash
  rkllama
```
