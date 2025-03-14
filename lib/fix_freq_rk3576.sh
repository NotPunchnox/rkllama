#!/system/bin/sh

# Check if debug mode is enabled (first argument)
DEBUG_MODE=${1:-0}

# Function for conditional echo based on debug mode
debug_echo() {
  if [ "$DEBUG_MODE" = "1" ]; then
    echo "$@"
  fi
}

# Disable CPU idle states to maximize performance
echo 1 > /sys/devices/system/cpu/cpu0/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu1/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu2/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu3/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu4/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu5/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu6/cpuidle/state1/disable
echo 1 > /sys/devices/system/cpu/cpu7/cpuidle/state1/disable

# NPU frequency management
debug_echo "NPU available frequencies:"
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/class/devfreq/27700000.npu/available_frequencies
fi
debug_echo "Fix NPU max frequency:"
echo userspace > /sys/class/devfreq/27700000.npu/governor
echo 1000000000 > /sys/class/devfreq/27700000.npu/userspace/set_freq
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/class/devfreq/27700000.npu/cur_freq
fi

# CPU frequency management
debug_echo "CPU available frequencies:"
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/devices/system/cpu/cpufreq/policy0/scaling_available_frequencies
  cat /sys/devices/system/cpu/cpufreq/policy4/scaling_available_frequencies
fi
debug_echo "Fix CPU max frequency:"
echo userspace > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor
echo 2208000 > /sys/devices/system/cpu/cpufreq/policy0/scaling_setspeed
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq
fi
echo userspace > /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
echo 2304000 > /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
fi

# GPU frequency management
debug_echo "GPU available frequencies:"
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/class/devfreq/27800000.gpu/cur_freq
  cat /sys/class/devfreq/27800000.gpu/available_frequencies
fi
debug_echo "Fix GPU max frequency:"
echo userspace > /sys/class/devfreq/27800000.gpu/governor
echo 950000000 > /sys/class/devfreq/27800000.gpu/userspace/set_freq
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/class/devfreq/27800000.gpu/cur_freq
fi

# DDR frequency management
debug_echo "DDR available frequencies:"
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/class/devfreq/dmc/available_frequencies
fi
debug_echo "Fix DDR max frequency:"
echo userspace > /sys/class/devfreq/dmc/governor
echo 2112000000 > /sys/class/devfreq/dmc/userspace/set_freq
if [ "$DEBUG_MODE" = "1" ]; then
  cat /sys/class/devfreq/dmc/cur_freq
fi

# When not in debug mode, just print a simple summary
if [ "$DEBUG_MODE" != "1" ]; then
  echo "RK3576 frequencies optimized for NPU inferencing"
fi