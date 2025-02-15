#!/bin/bash

# Extracting the CPU model if specific CPU model is not defined in env
if [ -z "$CPU_MODEL" ] && [ "${CPU_MODEL+xxx}" = "xxx" ]; then
  CPU_MODEL=$(cat /proc/cpuinfo | grep "cpu model" | head -n 1 | sed 's/.*Rockchip \(RK[0-9]*\).*/\1/' | tr '[:upper:]' '[:lower:]')
fi

# Starting the server with the CPU model as a parameter
python3 ~/RKLLAMA/server.py --target_platform "$CPU_MODEL"
