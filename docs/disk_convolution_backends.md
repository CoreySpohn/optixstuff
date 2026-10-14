# Disk convolution backends and the A100 experiment

Status: experimental, September 11, 2026. The Triton kernels have not been run
on a GPU yet. CPU checks establish the JAX integration and derivative rules;
they do not establish CUDA compilation, GPU numerical accuracy, or GPU speed.

## Recommendation

Keep **JAX as the default** and **Triton as an optional extra**. Offer explicit
backend selection on the table coronagraph. If the A100 experiment succeeds,
a later `auto` policy could select Triton for measured winning shapes, dtypes,
and devices while keeping a portable JAX fallback. Installing a GPU extra and
choosing a GPU runtime policy are separate decisions; even a future default
GPU policy should not make every CPU installation depend on Triton.

Upstream Triton currently lists Linux, NVIDIA compute capability 8.0+, and AMD
ROCm 6.2+ support, with CPU work under development. This prototype specifically
registers a CUDA lowering and does not claim AMD, Apple GPU, TPU, or CPU kernel
support. The A100 is an appropriate target. The separate Triton-CPU project
identifies itself as experimental; it is not a replacement for a supported
portable CPU backend today. Sources: [Triton compatibility](https://github.com/triton-lang/triton#compatibility),
[Triton-CPU](https://github.com/triton-lang/triton-cpu).

The optional extra pins the experiment's API family: JAX 0.10.x,
JAX-Triton 0.3.1, Triton 3.6.0. The Colab requirements pin JAX to 0.10.2.
Package resolution for Linux/Python 3.12 was checked locally; installation and
execution on Colab remain to be checked. Newer upstream versions exist, so
these pins are a reproducibility choice, not a claim to use the latest stack.
[JAX-Triton calling convention](https://jax-ml.github.io/jax-triton/triton_call/).

## Ownership and public selection

Yippy builds and loads PSFs. Optixstuff applies the spatially varying PSF
operator. Coronagraphoto converts disk contrast to photon rates and applies
its final nonnegative clipping. Photomancy supplies inference. The optional
kernel lives in Optixstuff so there is one optical implementation shared by
simulation and fitting.

```python
import equinox as eqx
from optixstuff import YippyCoronagraph

coro = YippyCoronagraph(
    yip_path,
    ensure_psf_datacube=True,
    disk_convolution_backend="triton",
)
# For an existing YippyCoronagraph inside an OpticalPath:
path = eqx.tree_at(
    lambda p: p.coronagraph.disk_convolution_backend, path, "jax-batched"
)
```

| Table setting | Execution | Derivatives |
|---|---|---|
| `jax` | Existing full/quarter einsum; unchanged default | Existing JAX behavior |
| `jax-batched` | Explicit candidate-by-source matrix multiplication | Ordinary JAX AD, including cube derivatives |
| `triton` | Experimental split-source CUDA reduction | Source JVP, VJP, higher derivatives; fixed cube |
| `reference` | Custom operator rules lowered to JAX | Tests the Triton derivative rules on CPU; fixed cube |

Use `eqx.filter_jit` with the model as an argument. Store the optical path in
an Equinox forward module, as in `benchmarks/photomancy_disk_fit.py`, and inject
it into `photomancy.disk.build_disk_logdensity`. This keeps the resident cube
as a dynamic array argument. A Python closure containing the cube can instead
turn it into a large compilation constant. Select the backend before compiling.
Custom table subclasses with their own constructor should assign the inherited
`disk_convolution_backend` field if they want to replace it via `eqx.tree_at`.

The lower-level `optixstuff.experimental.disk_convolution.convolve_disk` accepts
`(..., source_y, source_x)` candidates sharing one full or quarter cube. Its
`backend="jax"` is the flattened matrix path; the table's legacy default remains
separate. Triton rejects CPU use and derivatives/vmap over the cube rather than
silently falling back or returning incorrect optical derivatives. Map distinct
optical configurations with `jax.lax.map`, or use JAX. Device sharding and mixed
GPU configurations are outside this prototype.

## What the prototype computes

The optical model is

\[
I_{xy} = \sum_{ij} S_{ij} P_{ijxy}.
\]

Each source position has its own PSF. This is a dense linear operator, not a
translation-invariant convolution. Ordinary FFT convolution is generally a
change to the physical model.

The full cube is flattened without changing the source/output conventions.
Quarter cubes retain the existing fold/pad/flip convention, including its
specific odd/even center treatment; they are never expanded to a full cube.
The folds and output flips remain in JAX. Source rotation/resampling, detector
resampling, spectral and optical factors, and clipping also remain in the
existing simulation. The initial Triton kernels do not fuse these operations
or the likelihood; such fusion needs profiling evidence.

Both forward and transpose operations read the original resident PSF matrix.
The transpose uses strides, avoiding a stored transposed copy. Splitting the
source sum exposes parallelism for small candidate counts, followed by a JAX
sum of partial outputs. Scratch is capped near 64 MiB except when one output
slice itself exceeds that size. Small batches and FP64 use explicit
multiply/reduce tiles; larger FP32 batches use IEEE dot tiles. There is no FP16,
TF32, or atomic accumulation. This sacrifices some potential tensor-core
throughput until a scientific reduced-precision error budget exists.

JVP and transpose rules describe the fixed linear operator. They compose with
the nonlinear disk renderer, including Hessians required by Laplace inference.
Cube derivatives explicitly raise for the experimental primitive; this is a
fixed-optics disk-fitting experiment, not differentiable optical design.

## Running Colab

Open the local `disk_triton_colab.ipynb` in VS Code and select its Colab A100
kernel. In the local Explorer, right-click `disk-triton-colab.zip` and select
**Upload to Colab**, targeting the same server. The generated notebook and ZIP
live in `output/disk-triton/`. The first cell
extracts the uploaded ZIP and fails immediately if it is missing; no browser
upload widget is used. The bundle contains wheels of the
local first-party code, scripts, tests, requirements, and file hashes. No push,
release, or unpublished branch is needed. The notebook:

1. Installs the pinned stack and confirms it sees an A100.
2. Runs correctness tests. All sixteen CUDA-specific cases must execute and pass,
   including multiple source-reduction tiles in both forward and adjoint.
3. Benchmarks synthetic full/quarter operator workloads at B=1, 8, 32, in
   FP64 and FP32, with warm synchronized timings and separate first-call costs.
4. Fits a physical GraterDisk using a real, downsampled EAC1 AAVC PSF table,
   through coronagraphoto and Photomancy's Laplace backend.
5. Exports that cube for operator-only timings and packages compact JSON results.

The notebook stays local; Python and `/content` reside on the Colab server.
Results are written under `/content/disk-triton/output/`, while raw YIP downloads
and the exported PSF cube live under `/content/disk-triton/data/`. In VS Code's
**Colab > Contents** view, refresh and right-click the results ZIP to select
**Download...**. Save it into the project's local `output/disk-triton/` folder.
Save the local notebook to retain cell outputs. Download the cube separately,
or set `EXPORT_DATA = True` in the final cell to archive the entire data folder.
Raw YIP downloads are reused on the same server; this is not automatic local
file synchronization, and server recycling loses undownloaded files.
The extension also offers experimental server mounting for browsing the remote
filesystem in the workspace. See the [official VS Code extension guide](https://github.com/googlecolab/colab-vscode/wiki/User-Guide#working-with-files).

The real fit estimates radius, inclination, and position angle while fixing
other disk morphology parameters. It uses photon-rate images with fixed
Gaussian noise; it does not establish performance for Poisson likelihoods,
correlated residuals, all six disk parameters, or posterior sampling.
Inspect the final gradient norm before treating a fixed-step fit as converged.

Start at 64-pixel real PSFs. The native `(129, 129, 256, 256)` quarter cube is
4.36 GB in FP32 or 8.72 GB in FP64, before source arrays, derivatives, temporary
buffers, host copies, and JAX's allocator. Native evaluation and FP64 reference
comparisons need explicit host/device memory planning. Downsampling changes
sampling; agreement between backends on a coarse cube does not validate that
coarse approximation against the native optical model.

The fit script creates an isolated temporary YIP cache because Yippy's current
performance-cache filename does not distinguish downsampled PSF shapes. It also
uses a resolved 3 lambda/D aperture for the unused scalar performance tables;
this aperture does not enter disk convolution. A FITS native-byte-order fix in
Yippy is included so coarse PSFs can be built on a fresh Colab runtime.

## CPU optimization priorities

**First, change the contraction layout and measure the complete fit.** Flatten
candidate and quadrant axes into one matrix multiplication and keep all
instrument arrays dynamic. This is already available as `jax-batched`, with
ordinary JAX AD. Benchmark single proposals as well as batched proposals;
fitting cannot always provide a large shared-optics batch.

**Then profile the remaining work.** Measure source rendering, source/detector
resampling, the forward operator, transpose, and optimization separately.
Keep one compiled function per stable shape and dtype. Benchmark XLA/BLAS thread
settings on the target CPU rather than assuming all cores or more threads help.
Avoid host/device conversions inside repeated likelihood calls.

**Reduce exact work where the scientific setup permits it.** Precompute fixed
resampling weights and retain quarter symmetry. Restrict source/output grids
only when the excluded flux or likelihood contribution is demonstrably
negligible, including PSF wings. Fixed optical configurations can share work;
varying wavelength, roll, sampling, or fitted optical parameters limits reuse.

**Study low-rank compression next.** If `K` is well approximated by `U @ V.T`,
two smaller contractions accelerate CPU and GPU and admit a consistent
transpose. No rank or speedup is established here. Select rank by image,
noise-weighted likelihood, gradient, MAP/covariance, and evidence error on
held-out disk families, not only an unweighted matrix residual. Include setup
cost and the number of evaluations needed to amortize it.

Sparse/local PSF approximations and locally shift-invariant FFT models are
possible later studies. Neither is exact in general: faint PSF wings can matter
to high-contrast fitting. A hand-written CPU extension would add a second AD and
maintenance surface; pursue it only if profiling shows an important operation
that portable JAX/compiler optimizations cannot handle.

## Promotion criteria

CPU derivative tests are necessary but insufficient. Before enabling Triton by
default on any GPU: pass real GPU image/adjoint/gradient/Hessian/batching checks;
compare to FP64 in a declared noise budget; measure complete fits and uncertainty
or evidence; cover representative cube sizes, single/batched proposals, and
supported devices; and measure compilation, transfer, scratch, and peak memory.
Promote by measured regime, retaining explicit JAX selection. A custom dense
kernel may lose to XLA/vendor GEMM, especially for large batches.
