#!/bin/bash

# Extracting the CPU model
cpu_model=$(cat /proc/cpuinfo | grep "cpu model" | head -n 1 | sed 's/.*Rockchip \(RK[0-9]*\).*/\1/' | tr '[:upper:]' '[:lower:]')

# Starting the server with the CPU model as a parameter
python3 ~/RKLLAMA/server.py --target_platform "$cpu_model"
