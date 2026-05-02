<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary
-->

# PyTorch Profiler API Reference

## PyTorch 2.0+ Breaking Changes

When accessing profiler event attributes, use the **current PyTorch 2.0+ API**:

| Correct (PyTorch 2.0+) | Deprecated/Removed | Description |
|------------------------|-------------------|-------------|
| `device_time` | ~~`cuda_time`~~ | Total device time |
| `device_time_total` | ~~`cuda_time_total`~~ | Total device time (same as above) |
| `self_device_time_total` | ~~`self_cuda_time_total`~~ | Self device time excluding children |
| `cpu_time` | - | CPU time |
| `cpu_time_total` | - | Total CPU time |
| `self_cpu_time_total` | - | Self CPU time |

## Correct Usage

```python
for event in prof.key_averages():
    name = event.key
    cpu_time = event.cpu_time_total  # microseconds
    device_time = event.device_time_total  # microseconds (NOT cuda_time_total!)
    self_device_time = event.self_device_time_total  # NOT self_cuda_time_total!
```

## Common Mistakes

These attribute names raise `AttributeError` in PyTorch 2.0+:

- `event.cuda_time_total` — use `event.device_time_total` instead
- `event.self_cuda_time_total` — use `event.self_device_time_total` instead

## Sorting by Device Time

```python
# CORRECT
print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=20))

# WRONG (will error)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
```
