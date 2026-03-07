# src/gemv_implementations.jl

const WAVEFRONT_SIZE = 64

"""
    gemv_builtin(A::Matrix{Float16}, x::Vector{Float16}) -> Vector{Float32}

Built-in matrix-vector multiply after explicit promotion to Float32.

NOTE:
This is useful as a convenience/sanity implementation, but the exact
accumulation order may differ from the sequential and wavefront versions.
"""
function gemv_builtin(A::Matrix{Float16}, x::Vector{Float16})::Vector{Float32}
    M, K = size(A)
    @assert length(x) == K "Dimension mismatch: size(A) = $(size(A)), length(x) = $(length(x))"

    A32 = Float32.(A)
    x32 = Float32.(x)
    y = A32 * x32

    return Vector{Float32}(y)
end


"""
    gemv_sequential(A::Matrix{Float16}, x::Vector{Float16}) -> Vector{Float32}

Reference sequential GEMV:
- Float16 inputs
- convert each operand to Float32 before multiply
- accumulate in Float32
- left-to-right summation order
"""
function gemv_sequential(A::Matrix{Float16}, x::Vector{Float16})::Vector{Float32}
    M, K = size(A)
    @assert length(x) == K "Dimension mismatch: size(A) = $(size(A)), length(x) = $(length(x))"

    y = Vector{Float32}(undef, M)

    @inbounds for i in 1:M
        acc = 0.0f0
        for k in 1:K
            acc += Float32(A[i, k]) * Float32(x[k])
        end
        y[i] = acc
    end

    return y
end


"""
    gemv_wavefront(A::Matrix{Float16}, x::Vector{Float16}) -> Vector{Float32}

Models a 64-lane AMD wavefront:
- each lane accumulates a strided partial sum
- reduction is performed as an explicit binary tree:
    32, 16, 8, 4, 2, 1

This is the best CPU-side numerical model for the later GPU kernel.
"""
function gemv_wavefront(A::Matrix{Float16}, x::Vector{Float16})::Vector{Float32}
    M, K = size(A)
    @assert length(x) == K "Dimension mismatch: size(A) = $(size(A)), length(x) = $(length(x))"

    y = Vector{Float32}(undef, M)

    @inbounds for i in 1:M
        partials = zeros(Float32, WAVEFRONT_SIZE)

        # 1-based Julia indexing:
        # lane 1 -> k = 1, 65, 129, ...
        # lane 2 -> k = 2, 66, 130, ...
        for lane in 1:WAVEFRONT_SIZE
            acc = 0.0f0
            for k in lane:WAVEFRONT_SIZE:K
                acc += Float32(A[i, k]) * Float32(x[k])
            end
            partials[lane] = acc
        end

        # Explicit tree reduction to mirror GPU cross-lane reduction order
        offset = WAVEFRONT_SIZE ÷ 2
        while offset >= 1
            for lane in 1:offset
                partials[lane] += partials[lane + offset]
            end
            offset ÷= 2
        end

        y[i] = partials[1]
    end

    return y
end