# GPU Monitoring Commands

This document contains various commands for monitoring NVIDIA GPUs on Linux systems.

## Primary GPU Monitoring Commands

### 1. Basic GPU Status
```bash
nvidia-smi
```
Shows current GPU utilization, memory usage, temperature, and running processes.

### 2. Continuous Monitoring
```bash
nvidia-smi -l 1
```
Refreshes every 1 second (change the number to any interval you prefer).

### 3. Real-time Monitoring with Watch
```bash
watch -n 1 nvidia-smi
```
Updates the display every 1 second with a clean interface. Press `Ctrl+C` to exit.

### 4. Detailed Information in CSV Format
```bash
nvidia-smi --query-gpu=* --format=csv
```
Provides comprehensive GPU information in CSV format.

## Advanced Monitoring Options

### Continuous Monitoring with Specific Metrics
```bash
nvidia-smi -l 1 --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv
```

### Monitor GPU Processes
```bash
nvidia-smi pmon
```

### Monitor GPU Events
```bash
nvidia-smi dmon
```

## Quick Reference

| Command | Purpose |
|---------|---------|
| `nvidia-smi` | Basic GPU status |
| `watch -n 1 nvidia-smi` | Real-time monitoring (most common) |
| `nvidia-smi -l 1` | Continuous monitoring |
| `nvidia-smi pmon` | Process monitoring |
| `nvidia-smi dmon` | Event monitoring |

## Notes

- Replace `1` in the interval commands with your preferred refresh rate in seconds
- Use `Ctrl+C` to exit any continuous monitoring command
- The `watch` command provides the cleanest real-time interface
- All commands require NVIDIA drivers to be properly installed
