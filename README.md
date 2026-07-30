# cuda-playground

A set of CUDA kernels written from scratch to build up GPU performance intuition.
Every kernel is verified against a CPU reference and measured — the goal is not
"it runs", but "here is how fast it runs, and here is why".

This is a learning repository. The measurement habits established here
(CUDA-event timing, CPU reference verification, bandwidth-utilization reporting)
carry over into my larger GPU projects.

---

## Results

Measured on: **NVIDIA RTX 4070 Laptop** (see [`hardware.md`](hardware.md) for full specs)

| # | Kernel | Variant | Time (ms) | Effective BW (GB/s) | % of Peak | Bound by |
|---|--------|---------|-----------|---------------------|-----------|----------|
| 1 | Vector Add | naive | | | | memory |
| 1 | Vector Add | grid-stride | | | | memory |
| 2 | Vector Scale | out-of-place | | | | memory |
| 2 | Vector Scale | in-place | | | | memory |
| 3 | Element-wise | 3 separate launches | | | | memory |
| 3 | Element-wise | fused | | | | memory |

<!-- Problem size for the table above: N = ___ elements (fp32). Median of ___ runs after warm-up. -->

---

## Kernels

Each directory contains its own README with the full methodology, a size sweep,
and notes on what the numbers mean.

| # | Kernel | Focus | Notes |
|---|--------|-------|-------|
| 01 | [Vector Add](01-vector-add/) | Baseline, grid-stride loops | [notes](01-vector-add/README.md) |
| 02 | [Vector Scale](02-vector-scale/) | Read/write traffic, in-place vs out-of-place | [notes](02-vector-scale/README.md) |
| 03 | [Element-wise Ops](03-elementwise/) | Kernel fusion, launch overhead | [notes](03-elementwise/README.md) |

---

## Methodology

Consistency matters more than any single number, so every kernel follows the
same measurement protocol:

- **Timing:** CUDA events only. No CPU-side wall-clock timing of async work.
- **Warm-up:** first N iterations discarded (context init, clock ramp-up).
- **Reporting:** median of repeated runs, not the mean and not the best case.
- **Correctness:** every kernel is checked element-wise against a CPU reference
  with a fixed RNG seed before any timing result is recorded.
- **Effective bandwidth:** `(bytes read + bytes written) / elapsed time`,
  reported as a percentage of the theoretical peak in `hardware.md`.
- **Transfers excluded:** host↔device copies are timed separately and not folded
  into kernel numbers, unless a section explicitly says otherwise.

---

## Build

Requirements: CUDA Toolkit 12.8+, CMake 3.24+, a C++17 compiler.

```bash
cmake -B build -S .
cmake --build build --config Release
```

Built for `sm_89` (RTX 4070 Laptop) and `sm_120` (RTX 5080). Adjust
`CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` for other targets.

Run a single kernel benchmark:

```bash
./build/01-vector-add/vector_add
```

---

## Layout

```
cuda-playground/
├── common/            # shared timing + verification utilities
├── 01-vector-add/
├── 02-vector-scale/
├── 03-elementwise/
├── hardware.md        # device specs and theoretical peaks
└── README.md
```

---

## Log

<!-- One short entry per week: what was built, what was learned, what broke. -->

**Week 1 (2026-07-14):** _in progress_