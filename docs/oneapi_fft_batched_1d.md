# oneAPI.jl — batched 1D FFT primitive needed

**TL;DR.** `oneAPI.jl`'s FFT binding (`oneAPI.oneMKL.FFT`) only supports **full N-D FFT** — it rejects any `region` that doesn't cover every dimension of the input. There is no "batched 1D FFT" primitive (cuFFT's `cufftPlanMany`-equivalent). This blocks any Julia code that does `fft(x, dim)` along a single axis of a multi-dim oneArray. A real downstream use case (audio classification on Intel PVC GPUs via `PhasorNetworks.jl`) is forced into a CPU round-trip workaround that runs ~100× slower than the equivalent CUDA path.

This doc is the brief for whoever picks up the work to expose batched 1D FFT in `oneAPI.jl`. Self-contained — should not require reading any other repo to start.

---

## The ask

Expose a batched 1D FFT primitive in `oneAPI.jl` analogous to `CUDA.CUFFT`'s plan-many / strided-FFT support. Concretely: make this work, in-place or out-of-place, native (no host round-trip), differentiable via existing AbstractFFTs ChainRules:

```julia
using oneAPI, AbstractFFTs
x = oneArray(rand(ComplexF32, 64, 32_000, 16))
y = fft(x, 2)           # 1D FFT along dim 2, treating (dim 1, dim 3) as batch
```

Intel oneMKL itself supports this — see `DftiCreateDescriptor` with `DFTI_NUMBER_OF_TRANSFORMS` + `DFTI_INPUT_DISTANCE` / `DFTI_OUTPUT_DISTANCE`. The Julia wrapper just doesn't surface it yet.

---

## The constraint, observed

### Error A — non-leading region rejected

```julia
x = oneArray(rand(ComplexF32, 64, 32_000, 16))   # (C, N, B)
fft(x, 2)
# ERROR: Partial dimension FFT not yet supported. Region (2,) must be (1, 2)
```

### Error B — even leading-single-dim region is rejected

```julia
x = oneArray(rand(ComplexF32, 32_000, 64))      # (N, C)
fft(x, 1)
# ERROR: Partial dimension FFT not yet supported. Region (1,) must be (1, 2)
```

The pattern: for a D-dim array, the FFT region MUST be `(1, 2, ..., D)`. Any subset — leading prefix, trailing axis, single dim — is rejected. The error message is misleading (it says "must be (1, 2)" because the input is 2D; on a 3D input the message would be "must be (1, 2, 3)").

### What this implies about the binding

oneAPI.jl's FFT plan creation appears to forward the input shape directly to oneMKL DFT as a single-transform descriptor. oneMKL's DFTI supports *batched* transforms (multiple independent 1D FFTs over one axis) via a separate API path that the Julia wrapper hasn't exposed.

---

## The downstream use case (real and concrete)

`PhasorNetworks.jl` (https://github.com/wilkieolin/PhasorNetworks.jl) — a Julia neural network library for phase-aware spiking/oscillator-based computation. Its FFT-based causal convolution `causal_conv_fft` is the inner kernel of `ResonantSTFT`, which is the front-end of every audio classification architecture:

```julia
# src/kernels.jl:207
function causal_conv_fft(K::AbstractMatrix{<:Complex},
                         H::AbstractArray{<:Complex, 3})
    # ... K is (C, N), H is (C, N, B) ...
    K_f = fft(K_pad, 2)       # <-- needs batched 1D FFT
    H_f = fft(H_pad, 2)       # <-- needs batched 1D FFT
    # ... pointwise multiply ...
    ifft(Z_f, 2)              # <-- needs batched 1D IFFT
end
```

The shape pattern is **textbook batched 1D FFT**: one fixed length-N transform per `(channel, batch)` pair. Exactly what cuFFT's plan-many handles in one launch.

Without this primitive, every audio architecture using `ResonantSTFT` (which is the natural Phasor front-end for audio) is unrunnable natively on Intel GPUs.

---

## Shapes that matter to the user

Representative shapes from a realistic training workload (audio at 16 kHz, ~1 s clips, batchsize 16):

| Tensor | Shape | dtype | Bytes |
|---|---|---|---|
| `K_pad` | `(64, 32_000)` | `ComplexF32` | 16 MB |
| `H_pad` | `(64, 32_000, 16)` | `ComplexF32` | 256 MB |
| FFT axis length | 32_000 | — | — |
| Batch count (non-FFT axes) | 64 × 16 = 1024 | — | — |

Smaller shapes (smoke tests use batch=4, n_freqs=64) work fine via the workaround. The above is the smallest realistic training shape.

---

## What does NOT work (workarounds the user tried)

| Attempt | Why it fails |
|---|---|
| Permute time to leading dim then `fft(x, 1)` on 3D | Region `(1,)` on 3D array still rejected (must be `(1, 2, 3)`) |
| Reshape to 2D `(N, C·B)` then `fft(x, 1)` | Region `(1,)` on 2D still rejected (must be `(1, 2)`) |
| Reshape to 2D then full 2D `fft(x)` | Wrong math — FFTs along `C·B` axis too, mixing batch indices into the transform output |
| Insert trivial-length dims to pad shape | "Trivial" axis still occupies a dim that the full-region rule demands be FFT'd |
| Per-`(c, b)` loop over 1D `fft(x_1d)` | Each call passes the region check (full region on 1D array) — but kernel-launch overhead × 1024 transforms per call site × multiple call sites per forward × Zygote pullback makes it slower than the CPU round-trip; also confuses Zygote tracing through the loop |

There is no shape trick that produces a correct batched 1D FFT under the "region must cover all dims" rule.

---

## Current workaround (CPU round-trip)

```julia
function PhasorNetworks.causal_conv_fft(K, H::oneAPI.oneArray{<:Complex, 3})
    cpu = cpu_device()                          # Lux.cpu_device() — uses Adapt
    K_cpu = K |> cpu
    H_cpu = H |> cpu
    # ... cat + fft(_, 2) + multiply + ifft(_, 2) on CPU via FFTW ...
    return Z_cpu |> gdev_oneapi
end
```

Adapt's `ChainRules` rules carry the cotangent path through device transfers, so Zygote works without further patching.

### Performance cost (measured on Aurora, June 2026)

- **Smoke (batch=4, single fwd+bwd):** ~12 ms per `causal_conv_fft` call. All 5 SSM/SSA/SCA architectures pass.
- **Training (batch=16, sustained loop, real audio):** **6.6 s per fwd+bwd mini-batch**, dominated by host↔device transfer (~270 MB per call) plus single-threaded FFTW on the host. Per-epoch time on a single PVC tile: **~5.9 hours** for the simplest architecture (`stft_phasor_dense`).
- **Same architecture on NVIDIA GB10 (cuFFT, native):** roughly minutes per epoch. The CPU round-trip is **~100× slower** than what a native GPU FFT would give.

### Cascading failure mode (worth knowing)

When `fft(x, 2)` hits the oneAPI rejection, the GPU context is left in a corrupted state. The next allocation — even from a Julia `try` / `catch` recovery path — segfaults at the C driver level:

```
Segmentation fault from GPU at 0xff..., ctx_id: 1 (CCS) type: 0 (NotPresent), ...
Abort was called at 288 line in file: .../intel-compute-runtime-.../drm_neo.cpp
```

Julia's exception handling does not catch C-level driver aborts. Practical effect: a single failing FFT call can take out a long-running process. Worth documenting in the binding even before the fix lands.

---

## Pointers for the implementer

- **oneAPI.jl source:** look in `lib/mkl/dft.jl` and `src/fft.jl` (paths approximate; the FFT binding lives under the oneMKL wrapper hierarchy). The function `AbstractFFTs.plan_fft` for `oneArray` is the entry point that needs to handle non-trivial regions.
- **Intel oneMKL DFTI reference:** look up `DFTI_NUMBER_OF_TRANSFORMS`, `DFTI_INPUT_DISTANCE`, `DFTI_OUTPUT_DISTANCE`, `DFTI_INPUT_STRIDES`, `DFTI_OUTPUT_STRIDES`. The combination of `NUMBER_OF_TRANSFORMS` + distances/strides expresses batched 1D FFT.
- **Reference impl in cuFFT.jl:** `CUDA.CUFFT`'s `plan_fft` for `CuArray` already handles arbitrary regions via `cufftPlanMany` — useful for shape/stride logic, not for the actual oneMKL bindings.
- **AbstractFFTs.jl contract:** `plan_fft(x, region)` must produce a plan whose `*(plan, x)` matches `fft(x, region)` on CPU for all valid `region`. ChainRules for `fft` / `ifft` already exist via `AbstractFFTs`; no AD work needed on the implementer's side as long as the new code path is exercised through the same dispatch.

---

## Acceptance / verification

A minimal correctness test the implementer can run:

```julia
using oneAPI, AbstractFFTs, FFTW
for region in [(1,), (2,), (1, 2)]
    x = rand(ComplexF32, 64, 32_000, 16)
    y_cpu = fft(x, region)
    y_gpu = Array(fft(oneArray(x), region))
    @assert isapprox(y_cpu, y_gpu; rtol=1e-4) "mismatch on region=$region"
end
```

All three regions on the 3D input should pass. Currently only the full-region `(1, 2, 3)` case works.

A perf target: 1D FFT along dim 2 of `(64, 32_000, 16)` `ComplexF32` should run in <50 ms native on a single PVC tile (cuFFT does the same shape in ~10 ms on an A100; PVC HBM bandwidth is similar). The CPU round-trip is currently ~6 s, so the target gives ~100× headroom for the binding.

---

## Adjacent issue (out of scope for this brief)

`LuxLib` has no Intel-GPU `Conv` kernel. Its generic fallback iterates with scalar indexing, which `oneAPI.jl` blocks by default. Independent of this FFT issue, but worth flagging if the same person is auditing the Intel-GPU Julia stack: it blocks any neural net that uses 2D/4D `Conv` or `MaxPool` on Aurora.

---

## Source / attribution

- Observed on: Aurora compute nodes (Intel PVC GPUs), Julia 1.12.6, `libraries/julia/1.12` module, June 2026.
- Use case: `mos2_oscillators` project (https://github.com/wilkieolin — private), audio keyword-spotting classifier built on `PhasorNetworks.jl`.
- Reproducer scripts: `scripts/smoke_test_archs_aurora.jl` (synthetic smoke, all 5 SSM/SSA/SCA archs) and `scripts/train_audio_ssm_attention_aurora.jl` (real training).
- Workaround commit: see `causal_conv_fft` override in either of the above scripts.
