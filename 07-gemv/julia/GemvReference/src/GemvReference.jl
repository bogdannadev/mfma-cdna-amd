module GemvReference

using Random
using Printf
using Statistics

include("gemv_implementations.jl")
include("test_vectors.jl")
include("precision_analysis.jl")

greet() = print("Hello World!")

end # module GemvReference
