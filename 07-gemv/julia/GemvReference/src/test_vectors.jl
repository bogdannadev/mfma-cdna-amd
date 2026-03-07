# src/test_vectors.jl

struct TestCase
    name::String
    M::Int
    K::Int
    seed_offset::Int
end

const CASES = [
    TestCase("case01_M4_K4",       4,    4,    1),
    TestCase("case02_M256_K64",  256,   64,    2),
    TestCase("case03_M256_K4096",256, 4096,    3),
    TestCase("case04_M8192_K128",8192, 128,    4),
    TestCase("case05_M64_K8192",  64, 8192,    5),
    TestCase("case06_M4096_K4096",4096,4096,   6),
]

const DEFAULT_SEED = 42

"""
    generate_inputs(case::TestCase; seed::Int=DEFAULT_SEED)

Generate deterministic Float16 inputs in [-1, 1].
"""
function generate_inputs(case::TestCase; seed::Int=DEFAULT_SEED)
    rng = MersenneTwister(seed + case.seed_offset)

    A = Float16.(2.0f0 .* rand(rng, Float32, case.M, case.K) .- 1.0f0)
    x = Float16.(2.0f0 .* rand(rng, Float32, case.K) .- 1.0f0)

    return A, x
end


"""
    write_matrix_rowmajor(filepath, A)

Writes Julia column-major matrix A to disk in row-major layout expected by C/HIP.

Trick:
- Julia writes matrices in column-major memory order.
- If we transpose A first, then writing the transposed matrix's raw bytes
  produces the original A in row-major order on disk.
"""
function write_matrix_rowmajor(filepath::AbstractString, A::Matrix{Float16})
    At = permutedims(A, (2, 1))  # K x M copy

    open(filepath, "w") do io
        write(io, At)
    end

    expected_bytes = length(A) * sizeof(Float16)
    actual_bytes = filesize(filepath)
    @assert actual_bytes == expected_bytes "A.bin size mismatch: got $actual_bytes, expected $expected_bytes"

    return nothing
end


"""
    write_vector(filepath, v)

Writes a raw vector to disk.
"""
function write_vector(filepath::AbstractString, v::Vector{T}) where {T}
    open(filepath, "w") do io
        write(io, v)
    end

    expected_bytes = length(v) * sizeof(T)
    actual_bytes = filesize(filepath)
    @assert actual_bytes == expected_bytes "Vector file size mismatch: got $actual_bytes, expected $expected_bytes"

    return nothing
end


"""
    write_meta(filepath, M, K)

Writes:
M=<value>
K=<value>
"""
function write_meta(filepath::AbstractString, M::Int, K::Int)
    open(filepath, "w") do io
        println(io, "M=$M")
        println(io, "K=$K")
    end
    return nothing
end


"""
    read_vector(filepath, ::Type{T}, n) -> Vector{T}

Read raw binary vector of length n.
"""
function read_vector(filepath::AbstractString, ::Type{T}, n::Int) where {T}
    buf = Vector{T}(undef, n)
    open(filepath, "r") do io
        read!(io, buf)
    end
    return buf
end


"""
    read_matrix_rowmajor(filepath, M, K) -> Matrix{Float16}

Reads a row-major Float16 matrix from disk into a normal Julia column-major matrix A[M, K].
"""
function read_matrix_rowmajor(filepath::AbstractString, M::Int, K::Int)::Matrix{Float16}
    raw = read_vector(filepath, Float16, M * K)

    # raw is row-major A[i, k] laid out as rows.
    # Reinterpret as K x M column-major then transpose back.
    At = reshape(raw, K, M)          # this matches bytes written from permutedims(A, (2,1))
    A = permutedims(At, (2, 1))      # back to M x K

    return A
end


"""
    generate_case(case, base_dir; reference=:wavefront, seed=DEFAULT_SEED)

Generate one case directory with:
- A.bin
- x.bin
- y_ref.bin
- meta.txt

reference:
- :wavefront   -> recommended for GPU comparison
- :sequential
- :builtin
"""
function generate_case(case::TestCase, base_dir::AbstractString;
                       reference::Symbol = :wavefront,
                       seed::Int = DEFAULT_SEED)

    dir = joinpath(base_dir, case.name)
    mkpath(dir)

    A, x = generate_inputs(case; seed=seed)

    y_ref = if reference === :wavefront
        gemv_wavefront(A, x)
    elseif reference === :sequential
        gemv_sequential(A, x)
    elseif reference === :builtin
        gemv_builtin(A, x)
    else
        error("Unknown reference = $reference. Use :wavefront, :sequential, or :builtin.")
    end

    write_matrix_rowmajor(joinpath(dir, "A.bin"), A)
    write_vector(joinpath(dir, "x.bin"), x)
    write_vector(joinpath(dir, "y_ref.bin"), y_ref)
    write_meta(joinpath(dir, "meta.txt"), case.M, case.K)

    return (; dir, A, x, y_ref)
end


"""
    generate_all_cases(base_dir; reference=:wavefront, seed=DEFAULT_SEED)

Generate all predefined cases.
"""
function generate_all_cases(base_dir::AbstractString;
                            reference::Symbol = :wavefront,
                            seed::Int = DEFAULT_SEED)

    for case in CASES
        result = generate_case(case, base_dir; reference=reference, seed=seed)
        @printf("Generated %-18s  A[%d×%d] x[%d] -> y[%d]  (%s)\n",
                case.name, case.M, case.K, case.K, case.M, result.dir)
    end

    return nothing
end