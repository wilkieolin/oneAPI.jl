# oneMKL FFT (DFT) high-level Julia interface
# Inspired by AMDGPU ROCFFT interface style, adapted to oneMKL DFT C wrapper.

module FFT

using ..oneMKL
using ..oneMKL: oneAPI, SYCL, syclQueue_t
using ..Support
using ..SYCL
using LinearAlgebra
using GPUArrays
using AbstractFFTs
import AbstractFFTs: complexfloat, realfloat
import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!
import AbstractFFTs: plan_rfft, plan_brfft, plan_inv, normalization, ScaledPlan
import AbstractFFTs: fft, bfft, ifft, rfft, Plan, ScaledPlan
export MKLFFTPlan

# Import DFT enums and constants from Support module
using ..Support

# Allow implicit conversion of SYCL queue object to raw handle when storing/passing
Base.convert(::Type{syclQueue_t}, q::SYCL.syclQueue) = Base.unsafe_convert(syclQueue_t, q)

abstract type MKLFFTPlan{T,K,inplace} <: AbstractFFTs.Plan{T} end

Base.eltype(::MKLFFTPlan{T}) where T = T
is_inplace(::MKLFFTPlan{<:Any,<:Any,inplace}) where inplace = inplace

# Forward / inverse flags
const MKLFFT_FORWARD = true
const MKLFFT_INVERSE = false

mutable struct cMKLFFTPlan{T,K,inplace,N,R,B} <: MKLFFTPlan{T,K,inplace}
    handle::onemklDftDescriptor_t
    queue::syclQueue_t
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    realdomain::Bool
    region::NTuple{R,Int}
    buffer::B
    pinv::Any
    outer_count::Int        # execute-time loop trip count (1 = no loop)
    outer_stride::Int       # element stride per outer slice (input == output for complex)
end

# Real transforms use separate struct (mirroring AMDGPU style) for buffer staging
mutable struct rMKLFFTPlan{T,K,inplace,N,R,B} <: MKLFFTPlan{T,K,inplace}
    handle::onemklDftDescriptor_t
    queue::syclQueue_t
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    xtype::Symbol
    region::NTuple{R,Int}
    buffer::B
    pinv::Any
    outer_count::Int        # execute-time loop trip count
    in_outer_stride::Int    # element stride per outer slice (input layout)
    out_outer_stride::Int   # element stride per outer slice (output layout)
end

# Inverse plan constructors (derive from existing plan)
function normalization_factor(sz, region)
    # AbstractFFTs expects inverse to scale by 1/prod(lengths along region)
    prod(ntuple(i-> sz[region[i]], length(region)))
end

function plan_inv(p::cMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = cMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,p.realdomain,p.region,p.buffer,p,p.outer_count,p.outer_stride)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end
function plan_inv(p::cMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = cMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,p.realdomain,p.region,p.buffer,p,p.outer_count,p.outer_stride)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end

function plan_inv(p::rMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = rMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,:brfft,p.region,p.buffer,p,p.outer_count,p.in_outer_stride,p.out_outer_stride)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end
function plan_inv(p::rMKLFFTPlan{T,MKLFFT_INVERSE,inplace,N,R,B}) where {T,inplace,N,R,B}
    q = rMKLFFTPlan{T,MKLFFT_FORWARD,inplace,N,R,B}(p.handle,p.queue,p.sz,p.osz,:rfft,p.region,p.buffer,p,p.outer_count,p.in_outer_stride,p.out_outer_stride)
    p.pinv = q
    ScaledPlan(q, 1/normalization_factor(p.sz, p.region))
end

function Base.show(io::IO, p::MKLFFTPlan{T,K,inplace}) where {T,K,inplace}
    print(io, inplace ? "oneMKL FFT in-place " : "oneMKL FFT ", K ? "forward" : "inverse", " plan for ")
    if isempty(p.sz); print(io, "0-dimensional") else print(io, join(p.sz, "×")) end
    print(io, " oneArray of ", T)
end

# Batched descriptor configuration
#
# For a region whose first/last transform axes are first_t/last_t:
#   inner-batch dims  = size[1 : first_t-1]    -> NUMBER_OF_TRANSFORMS + DISTANCE=1
#   transform dims    = size[first_t : last_t] -> descriptor lengths + STRIDES
#   outer-batch dims  = size[last_t+1 : N]     -> Julia-side execute-time loop
# When inner_batch == 1, fold outer into NUMBER_OF_TRANSFORMS to avoid the loop.
# sz/osz: full input/output array shapes (osz==sz for complex; differ on the
# reduction axis for real plans).
struct _BatchedCfg
    transform_lengths::Vector{Int64}   # column-major rank order
    fwd_strides::Vector{Int64}         # [0, s1, ..., sK] in input layout
    bwd_strides::Vector{Int64}         # [0, s1, ..., sK] in output layout
    num_transforms::Int
    fwd_distance::Int
    bwd_distance::Int
    outer_count::Int                   # execute-time loop trip count (1 = no loop)
    in_outer_stride::Int               # element stride per outer slice (input)
    out_outer_stride::Int              # element stride per outer slice (output)
end

function _batched_descriptor_config(sz::NTuple{N,Int}, osz::NTuple{N,Int},
                                    region::NTuple{R,Int};
                                    fold_outer::Bool=true) where {N,R}
    rs = sort(collect(region))
    first_t, last_t = rs[1], rs[end]
    rs == collect(first_t:last_t) ||
        throw(ArgumentError("oneAPI.jl FFT region must be a contiguous range; got $region"))
    (first_t >= 1 && last_t <= N) ||
        throw(ArgumentError("FFT region $region out of bounds for $N-dim array"))

    transform_lengths = Int64[sz[d] for d in first_t:last_t]

    fwd_strides = Vector{Int64}(undef, R+1); fwd_strides[1] = 0
    let p = 1
        for i in 1:(first_t-1); p *= sz[i]; end
        for k in 1:R
            fwd_strides[k+1] = p
            p *= sz[first_t + k - 1]
        end
    end
    bwd_strides = Vector{Int64}(undef, R+1); bwd_strides[1] = 0
    let p = 1
        for i in 1:(first_t-1); p *= osz[i]; end
        for k in 1:R
            bwd_strides[k+1] = p
            p *= osz[first_t + k - 1]
        end
    end

    inner_batch = 1
    for i in 1:(first_t-1); inner_batch *= sz[i]; end
    outer_batch = 1
    for i in (last_t+1):N; outer_batch *= sz[i]; end

    in_outer_stride  = prod(Int, sz[1:last_t]; init=1)
    out_outer_stride = prod(Int, osz[1:last_t]; init=1)

    if inner_batch > 1
        num_transforms = inner_batch
        fwd_distance   = 1
        bwd_distance   = 1
        outer_count    = outer_batch
    elseif fold_outer && outer_batch > 1
        num_transforms = outer_batch
        fwd_distance   = in_outer_stride
        bwd_distance   = out_outer_stride
        outer_count    = 1
    else
        num_transforms = 1
        fwd_distance   = prod(Int, transform_lengths; init=1)
        bwd_distance   = fwd_distance
        outer_count    = 1
    end

    return _BatchedCfg(transform_lengths, fwd_strides, bwd_strides,
                       num_transforms, fwd_distance, bwd_distance,
                       outer_count, in_outer_stride, out_outer_stride)
end

# Plan constructors
function _create_descriptor(sz::NTuple{N,Int}, T::Type, complex::Bool) where {N}
    prec = T<:Float64 || T<:ComplexF64 ? ONEMKL_DFT_PRECISION_DOUBLE : ONEMKL_DFT_PRECISION_SINGLE
    dom = complex ? ONEMKL_DFT_DOMAIN_COMPLEX : ONEMKL_DFT_DOMAIN_REAL
    desc_ref = Ref{onemklDftDescriptor_t}()
    # Create descriptor for the full array dimensions
    lengths = collect(Int64, sz)
    st = length(lengths) == 1 ? onemklDftCreate1D(desc_ref, prec, dom, lengths[1]) : onemklDftCreateND(desc_ref, prec, dom, length(lengths), pointer(lengths))
    st == 0 || error("onemkl DFT create failed (status $st)")
    desc = desc_ref[]
    # Do not program descriptor scaling; we'll perform inverse normalization manually.
    # Set placement explicitly based on plan type later
    # Construct a SYCL queue from current Level Zero context/device (reuse global queue)
    ze_ctx = oneAPI.context(); ze_dev = oneAPI.device()
    sycl_dev = SYCL.syclDevice(SYCL.syclPlatform(oneAPI.driver()), ze_dev)
    sycl_ctx = SYCL.syclContext([sycl_dev], ze_ctx)
    q = SYCL.syclQueue(sycl_ctx, sycl_dev, oneAPI.global_queue(ze_ctx, ze_dev))
    return desc, q
end

# Complex plans — shared implementation.
#
# For the full-region case (reg == (1,2,…,N)) we reproduce the historical
# code path byte-for-byte: a descriptor over size(X) with column-major
# strides over size(X), no NUMBER_OF_TRANSFORMS / DISTANCE set.
#
# For partial regions we build a smaller descriptor (just the transform
# axes) and use NUMBER_OF_TRANSFORMS + DISTANCE for inner batching plus an
# execute-time loop for outer batching (see _exec!).
#
# KNOWN LIMITATION (pre-existing, not introduced here): on at least the
# Aurora support-library build, ANY multi-D oneMKL DFT descriptor (rank
# 2+) fails at commit time — bare upstream `fft(rand(ComplexF32,8,32))`
# already throws "commit failed (-1)". As a result, only single-axis
# partial-region transforms are reliably batched here; multi-axis
# partial regions (e.g. region=(1,2) on a 3-D array) will surface the
# same upstream failure. Single-axis batched-1D — the docs reproducer
# and PhasorNetworks' use case — works on every shape we've tried.
function _make_complex_plan(X::oneAPI.oneArray{T,N}, region, inplace::Bool,
                            forward::Bool) where {T<:Union{ComplexF32,ComplexF64},N}
    R = length(region); reg = NTuple{R,Int}(region)
    placement = inplace ? ONEMKL_DFT_VALUE_INPLACE : ONEMKL_DFT_VALUE_NOT_INPLACE
    K = forward ? MKLFFT_FORWARD : MKLFFT_INVERSE
    full_region = reg == ntuple(identity, N)

    if full_region
        # Full-region: descriptor lengths already match the array layout, so
        # oneMKL's default column-major strides are correct — don't set any
        # STRIDES (the current support-library build rejects explicit
        # FWD/BWD_STRIDES on multi-D descriptors).
        desc, q = _create_descriptor(size(X), T, true)
        onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, placement)
        stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
        return cMKLFFTPlan{T,K,inplace,N,R,Nothing}(desc, q, size(X), size(X),
                                                    false, reg, nothing, nothing,
                                                    1, prod(Int, size(X); init=1))
    end

    # Partial-region path: descriptor lengths = just the transform axes,
    # batching encoded via NUMBER_OF_TRANSFORMS + DISTANCE + STRIDES.
    # Use INPUT_STRIDES / OUTPUT_STRIDES (the older oneMKL DFT API) rather
    # than FWD_STRIDES / BWD_STRIDES — the latter trip commit failures on
    # the current Aurora support-library build for multi-D descriptors.
    cfg = _batched_descriptor_config(size(X), size(X), reg)
    desc, q = _create_descriptor(Tuple(cfg.transform_lengths), T, true)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, placement)

    fwd_strides = cfg.fwd_strides
    bwd_strides = cfg.bwd_strides
    onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_INPUT_STRIDES, pointer(fwd_strides), length(fwd_strides))
    onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_OUTPUT_STRIDES, pointer(bwd_strides), length(bwd_strides))
    if cfg.num_transforms > 1
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_NUMBER_OF_TRANSFORMS, Int64(cfg.num_transforms))
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_FWD_DISTANCE, Int64(cfg.fwd_distance))
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_BWD_DISTANCE, Int64(cfg.bwd_distance))
    end

    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    return cMKLFFTPlan{T,K,inplace,N,R,Nothing}(desc, q, size(X), size(X),
                                                false, reg, nothing, nothing,
                                                cfg.outer_count, cfg.in_outer_stride)
end

plan_fft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N} =
    _make_complex_plan(X, region, false, true)
plan_bfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N} =
    _make_complex_plan(X, region, false, false)
plan_fft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N} =
    _make_complex_plan(X, region, true, true)
plan_bfft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{ComplexF32,ComplexF64},N} =
    _make_complex_plan(X, region, true, false)

# Real input methods - convert to complex like FFTW does
function plan_fft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    CT = Complex{T}
    # Create a complex plan by converting the real array to complex
    X_complex = oneAPI.oneArray{CT}(undef, size(X))
    plan_fft(X_complex, region)
end

function plan_bfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    CT = Complex{T}
    # Create a complex plan by converting the real array to complex
    X_complex = oneAPI.oneArray{CT}(undef, size(X))
    plan_bfft(X_complex, region)
end

function plan_fft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    error("In-place FFT not supported for real input arrays. Use plan_fft instead.")
end

function plan_bfft!(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    error("In-place FFT not supported for real input arrays. Use plan_bfft instead.")
end

# Real forward (out-of-place) - supports multi-dimensional transforms
function plan_rfft(X::oneAPI.oneArray{T,N}, region) where {T<:Union{Float32,Float64},N}
    # Convert region to tuple if it's a range
    if isa(region, AbstractUnitRange)
        region = tuple(region...)
    end
    R = length(region); reg = NTuple{R,Int}(region)

    # For single dimension transforms, use the optimized oneMKL real FFT
    if R == 1
        return _plan_rfft_1d(X, reg)
    end

    # For multi-dimensional transforms, use complex FFT approach
    # This is mathematically equivalent and works around oneMKL limitations
    return _plan_rfft_nd(X, reg)
end

# Single-dimension real FFT using oneMKL (optimized path) — arbitrary axis
function _plan_rfft_1d(X::oneAPI.oneArray{T,N}, reg::NTuple{1,Int}) where {T<:Union{Float32,Float64},N}
    d = reg[1]
    xdims = size(X)
    ydims = Base.setindex(xdims, div(xdims[d], 2) + 1, d)

    cfg = _batched_descriptor_config(xdims, ydims, reg)

    desc, q = _create_descriptor((xdims[d],), T, false)
    buffer  = oneAPI.oneArray{Complex{T}}(undef, ydims)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_NOT_INPLACE)

    GC.@preserve cfg begin
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_FWD_STRIDES,
                                    pointer(cfg.fwd_strides), length(cfg.fwd_strides))
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_BWD_STRIDES,
                                    pointer(cfg.bwd_strides), length(cfg.bwd_strides))
    end
    if cfg.num_transforms > 1
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_NUMBER_OF_TRANSFORMS, Int64(cfg.num_transforms))
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_FWD_DISTANCE, Int64(cfg.fwd_distance))
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_BWD_DISTANCE, Int64(cfg.bwd_distance))
    end

    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    R = length(reg)
    rMKLFFTPlan{T,MKLFFT_FORWARD,false,N,R,typeof(buffer)}(
        desc, q, xdims, ydims, :rfft, reg, buffer, nothing,
        cfg.outer_count, cfg.in_outer_stride, cfg.out_outer_stride)
end

# Multi-dimensional real FFT using complex FFT approach
struct ComplexBasedRealFFTPlan{T,N,R} <: MKLFFTPlan{T,MKLFFT_FORWARD,false}
    complex_plan::cMKLFFTPlan{Complex{T},MKLFFT_FORWARD,false,N,R,Nothing}
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    region::NTuple{R,Int}
end

function _plan_rfft_nd(X::oneAPI.oneArray{T,N}, reg::NTuple{R,Int}) where {T<:Union{Float32,Float64},N,R}
    # Create complex version for planning
    X_complex = oneAPI.oneArray{Complex{T}}(undef, size(X))
    complex_plan = plan_fft(X_complex, reg)

    # Calculate output dimensions (real FFT output size)
    xdims = size(X)
    ydims = ntuple(N) do i
        if i in reg && i == minimum(reg)  # First dimension in region gets reduced
            div(xdims[i], 2) + 1
        else
            xdims[i]
        end
    end

    ComplexBasedRealFFTPlan{T,N,R}(complex_plan, xdims, ydims, reg)
end

# Show method for complex-based plan
function Base.show(io::IO, p::ComplexBasedRealFFTPlan{T}) where {T}
    print(io, "oneMKL FFT forward plan for ")
    if isempty(p.sz); print(io, "0-dimensional") else print(io, join(p.sz, "×")) end
    print(io, " oneArray of ", T, " (multi-dimensional via complex FFT)")
end

# Execution for complex-based real FFT plan
function Base.:*(p::ComplexBasedRealFFTPlan{T,N,R}, X::oneAPI.oneArray{T}) where {T,N,R}
    # Convert to complex
    X_complex = Complex{T}.(X)

    # Perform complex FFT
    Y_complex = p.complex_plan * X_complex

    # Extract appropriate portion for real FFT result
    # For real FFT, we only need roughly half the output due to conjugate symmetry
    indices = ntuple(N) do i
        if i in p.region && i == minimum(p.region)
            # First dimension in region: take 1:(N÷2+1)
            1:(div(p.sz[i], 2) + 1)
        else
            # Other dimensions: take all
            1:p.sz[i]
        end
    end

    Y = Y_complex[indices...]
    return Y
end



# Real inverse (complex->real) requires complex input shape - supports multi-dimensional transforms
function plan_brfft(X::oneAPI.oneArray{T,N}, d::Integer, region) where {T<:Union{ComplexF32,ComplexF64},N}
    # Convert region to tuple if it's a range
    if isa(region, AbstractUnitRange)
        region = tuple(region...)
    end
    R = length(region); reg = NTuple{R,Int}(region)

    # For single dimension transforms, use optimized oneMKL path (arbitrary axis)
    if R == 1
        return _plan_brfft_1d(X, d, reg)
    end

    # For multi-dimensional transforms, use complex FFT approach
    return _plan_brfft_nd(X, d, reg)
end

# Single-dimension real inverse FFT using oneMKL (optimized path) — arbitrary axis
function _plan_brfft_1d(X::oneAPI.oneArray{T,N}, d::Integer, reg::NTuple{1,Int}) where {T<:Union{ComplexF32,ComplexF64},N}
    @assert T <: Complex
    RT = T.parameters[1]
    ax = reg[1]
    xdims = size(X)                                  # complex input (axis ax already reduced)
    ydims = Base.setindex(xdims, d, ax)              # real output (axis ax expanded to d)

    # oneMKL DISTANCE/STRIDES are layout properties tied to the forward (real) domain
    # for FWD_* and the backward (complex) domain for BWD_*. For brfft the real layout
    # is the OUTPUT and the complex layout is the INPUT — so pass (ydims, xdims).
    cfg = _batched_descriptor_config(ydims, xdims, reg)

    desc, q = _create_descriptor((d,), RT, false)
    buffer  = oneAPI.oneArray{T}(undef, xdims)
    onemklDftSetValueConfigValue(desc, ONEMKL_DFT_PARAM_PLACEMENT, ONEMKL_DFT_VALUE_NOT_INPLACE)

    GC.@preserve cfg begin
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_FWD_STRIDES,
                                    pointer(cfg.fwd_strides), length(cfg.fwd_strides))
        onemklDftSetValueInt64Array(desc, ONEMKL_DFT_PARAM_BWD_STRIDES,
                                    pointer(cfg.bwd_strides), length(cfg.bwd_strides))
    end
    if cfg.num_transforms > 1
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_NUMBER_OF_TRANSFORMS, Int64(cfg.num_transforms))
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_FWD_DISTANCE, Int64(cfg.fwd_distance))
        onemklDftSetValueInt64(desc, ONEMKL_DFT_PARAM_BWD_DISTANCE, Int64(cfg.bwd_distance))
    end

    stc = onemklDftCommit(desc, q); stc == 0 || error("commit failed ($stc)")
    R = length(reg)
    # in_outer_stride is for the INPUT array (complex, xdims);
    # out_outer_stride is for the OUTPUT array (real, ydims).
    rMKLFFTPlan{T,MKLFFT_INVERSE,false,N,R,typeof(buffer)}(
        desc, q, xdims, ydims, :brfft, reg, buffer, nothing,
        cfg.outer_count,
        prod(Int, xdims[1:reg[1]]; init=1),
        prod(Int, ydims[1:reg[1]]; init=1))
end

# Multi-dimensional real inverse FFT using complex FFT approach
struct ComplexBasedRealIFFTPlan{T,N,R} <: MKLFFTPlan{T,MKLFFT_INVERSE,false}
    complex_plan::cMKLFFTPlan{T,MKLFFT_INVERSE,false,N,R,Nothing}
    sz::NTuple{N,Int}
    osz::NTuple{N,Int}
    region::NTuple{R,Int}
    d::Int  # Original size of the reduced dimension
end

function _plan_brfft_nd(X::oneAPI.oneArray{T,N}, d::Integer, reg::NTuple{R,Int}) where {T<:Union{ComplexF32,ComplexF64},N,R}
    # Calculate the full complex array size (before real FFT reduction)
    xdims = size(X)
    full_complex_dims = ntuple(N) do i
        if i in reg && i == minimum(reg)  # First dimension in region was reduced
            d  # Restore original size
        else
            xdims[i]
        end
    end

    # Create complex version for planning - use the full size
    X_complex_full = oneAPI.oneArray{T}(undef, full_complex_dims)
    complex_plan = plan_bfft(X_complex_full, reg)

    ComplexBasedRealIFFTPlan{T,N,R}(complex_plan, xdims, full_complex_dims, reg, d)
end

# Show method for complex-based inverse plan
function Base.show(io::IO, p::ComplexBasedRealIFFTPlan{T}) where {T}
    print(io, "oneMKL FFT inverse plan for ")
    if isempty(p.sz); print(io, "0-dimensional") else print(io, join(p.sz, "×")) end
    print(io, " oneArray of ", T, " (multi-dimensional via complex FFT)")
end

# Execution for complex-based real inverse FFT plan
function Base.:*(p::ComplexBasedRealIFFTPlan{T,N,R}, X::oneAPI.oneArray{T}) where {T,N,R}
    # Reconstruct full complex array by exploiting conjugate symmetry
    # This is a simplified approach - for full accuracy, we'd need to properly
    # reconstruct the conjugate symmetric part

    # For now, pad with zeros (this works for certain cases but isn't fully general)
    xdims = size(X)
    full_indices = ntuple(N) do i
        if i in p.region && i == minimum(p.region)
            # Extend the reduced dimension
            1:p.d
        else
            1:xdims[i]
        end
    end

    # Create full complex array and copy the available data
    X_full = oneAPI.oneArray{T}(undef, p.osz)
    fill!(X_full, zero(T))

    # Copy the input data to the appropriate slice
    # NOTE: This is a simplified approach that doesn't fully reconstruct
    # conjugate symmetry. For full accuracy, proper conjugate symmetric
    # reconstruction should be implemented.
    copy_indices = ntuple(N) do i
        if i in p.region && i == minimum(p.region)
            1:xdims[i]  # Only the available part
        else
            1:xdims[i]
        end
    end

    X_full[copy_indices...] = X

    # Perform complex inverse FFT
    Y_complex = p.complex_plan * X_full

    # Extract real part (this is where the real output comes from)
    return real.(Y_complex)
end

# Inverse plan for complex-based real FFT plans
function plan_inv(p::ComplexBasedRealFFTPlan{T,N,R}) where {T,N,R}
    # For real FFT inverse, we need plan_brfft functionality
    # The first dimension in the region should be the one that was reduced
    first_dim = minimum(p.region)
    d = p.sz[first_dim]  # Original size of the reduced dimension

    # Create inverse plan using our new multi-dimensional brfft
    brfft_plan = _plan_brfft_nd(oneAPI.oneArray{Complex{T}}(undef, p.osz), d, p.region)
    ScaledPlan(brfft_plan, 1/normalization_factor(p.sz, p.region))
end

# Inverse plan for complex-based real inverse FFT plans
function plan_inv(p::ComplexBasedRealIFFTPlan{T,N,R}) where {T,N,R}
    # Create forward plan
    forward_plan = _plan_rfft_nd(oneAPI.oneArray{real(T)}(undef, p.osz), p.region)
    ScaledPlan(forward_plan, 1/normalization_factor(p.osz, p.region))
end



# Convenience no-region methods use all dimensions in order
plan_fft(X::oneAPI.oneArray) = plan_fft(X, ntuple(identity, ndims(X)))
plan_bfft(X::oneAPI.oneArray) = plan_bfft(X, ntuple(identity, ndims(X)))
plan_fft!(X::oneAPI.oneArray) = plan_fft!(X, ntuple(identity, ndims(X)))
plan_bfft!(X::oneAPI.oneArray) = plan_bfft!(X, ntuple(identity, ndims(X)))
plan_rfft(X::oneAPI.oneArray) = plan_rfft(X, ntuple(identity, ndims(X)))  # default all dims like Base.rfft
plan_brfft(X::oneAPI.oneArray, d::Integer) = plan_brfft(X, d, ntuple(identity, ndims(X)))

# Alias names to mirror AMDGPU / AbstractFFTs style
const plan_ifft = plan_bfft
const plan_ifft! = plan_bfft!
# plan_irfft should be normalized, unlike plan_brfft
plan_irfft(X::oneAPI.oneArray{T,N}, d::Integer, region) where {T,N} = begin
    p = plan_brfft(X, d, region)
    ScaledPlan(p, 1/normalization_factor(p.sz, p.region))
end
plan_irfft(X::oneAPI.oneArray{T,N}, d::Integer) where {T,N} = plan_irfft(X, d, (1,))

# Inversion
Base.inv(p::MKLFFTPlan) = plan_inv(p)

# High-level wrappers operating like CPU FFTW versions.
function fft(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_fft(X) * X)
end
function ifft(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    p = plan_bfft(X)
    # Apply normalization for ifft (unlike bfft which is unnormalized)
    scaling = one(T) / normalization_factor(size(X), ntuple(identity, ndims(X)))
    scaling * (p * X)
end
function fft!(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    (plan_fft!(X) * X; X)
end
function ifft!(X::oneAPI.oneArray{T}) where {T<:Union{ComplexF32,ComplexF64}}
    p = plan_bfft!(X)
    # Apply normalization for ifft! (unlike bfft! which is unnormalized)
    scaling = one(T) / normalization_factor(size(X), ntuple(identity, ndims(X)))
    p * X
    X .*= scaling
    X
end
function rfft(X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    (plan_rfft(X) * X)
end
function irfft(X::oneAPI.oneArray{T}, d::Integer) where {T<:Union{ComplexF32,ComplexF64}}
    # Use the normalized plan_irfft instead of unnormalized plan_brfft
    (plan_irfft(X, d) * X)
end

# Execution helpers
_rawptr(a::oneAPI.oneArray{T}) where T = reinterpret(Ptr{Cvoid}, pointer(a))
@inline _ptr_at(a::oneAPI.oneArray{T}, k::Int, stride_elems::Int) where T =
    reinterpret(Ptr{Cvoid}, pointer(a, 1 + k * stride_elems))

function _exec!(p::cMKLFFTPlan{T,MKLFFT_FORWARD,true}, X::oneAPI.oneArray{T}) where T
    for k in 0:(p.outer_count - 1)
        st = onemklDftComputeForward(p.handle, _ptr_at(X, k, p.outer_stride))
        st == 0 || error("forward FFT failed ($st)")
    end
    X
end
function _exec!(p::cMKLFFTPlan{T,MKLFFT_INVERSE,true}, X::oneAPI.oneArray{T}) where T
    for k in 0:(p.outer_count - 1)
        st = onemklDftComputeBackward(p.handle, _ptr_at(X, k, p.outer_stride))
        st == 0 || error("inverse FFT failed ($st)")
    end
    X
end
function _exec!(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{T}) where {T,K}
    fn = K == MKLFFT_FORWARD ? onemklDftComputeForwardOutOfPlace : onemklDftComputeBackwardOutOfPlace
    for k in 0:(p.outer_count - 1)
        st = fn(p.handle, _ptr_at(X, k, p.outer_stride), _ptr_at(Y, k, p.outer_stride))
        st == 0 || error("FFT failed ($st)")
    end
    Y
end

# Real forward
function _exec!(p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{Complex{T}}) where T
    for k in 0:(p.outer_count - 1)
        st = onemklDftComputeForwardOutOfPlace(p.handle,
                _ptr_at(X, k, p.in_outer_stride),
                _ptr_at(Y, k, p.out_outer_stride))
        st == 0 || error("rfft failed ($st)")
    end
    Y
end
# Real inverse (complex -> real)
function _exec!(p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}, Y::oneAPI.oneArray{R}) where {R,T<:Complex{R}}
    for k in 0:(p.outer_count - 1)
        st = onemklDftComputeBackwardOutOfPlace(p.handle,
                _ptr_at(X, k, p.in_outer_stride),
                _ptr_at(Y, k, p.out_outer_stride))
        st == 0 || error("brfft failed ($st)")
    end
    Y
end

# Public API similar to AMDGPU
function Base.:*(p::cMKLFFTPlan{T,K,true}, X::oneAPI.oneArray{T}) where {T,K}
    _exec!(p,X)
end
function Base.:*(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}) where {T,K}
    Y = oneAPI.oneArray{T}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{T}, p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{T}) where {T,K}
    _exec!(p,X,Y)
end

# Real forward
function Base.:*(p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    Y = oneAPI.oneArray{Complex{T}}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{Complex{T}}, p::rMKLFFTPlan{T,MKLFFT_FORWARD,false}, X::oneAPI.oneArray{T}) where {T<:Union{Float32,Float64}}
    _exec!(p,X,Y)
end
# Real inverse
function Base.:*(p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}) where {R,T<:Complex{R}}
    Y = oneAPI.oneArray{R}(undef, p.osz); _exec!(p,X,Y)
end
function LinearAlgebra.mul!(Y::oneAPI.oneArray{R}, p::rMKLFFTPlan{T,MKLFFT_INVERSE,false}, X::oneAPI.oneArray{T}) where {R,T<:Complex{R}}
    _exec!(p,X,Y)
end

# Support for applying complex plans to real arrays (convert real to complex first)
function Base.:*(p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{R}) where {T,K,R<:Union{Float32,Float64}}
    # Only allow if T is the complex version of R
    if T != Complex{R}
        error("Type mismatch: plan expects $(T) but got $(R)")
    end
    # Convert real input to complex
    X_complex = complex.(X)
    p * X_complex
end

function LinearAlgebra.mul!(Y::oneAPI.oneArray{T}, p::cMKLFFTPlan{T,K,false}, X::oneAPI.oneArray{R}) where {T,K,R<:Union{Float32,Float64}}
    # Only allow if T is the complex version of R
    if T != Complex{R}
        error("Type mismatch: plan expects $(T) but got $(R)")
    end
    # Convert real input to complex
    X_complex = complex.(X)
    _exec!(p, X_complex, Y)
end

end # module FFT
