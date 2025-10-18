using Finch
using MatrixDepot
using BenchmarkTools
using SparseArrays

include("fw-base.jl")
include("fw-base-adv.jl")
include("fw-as-1d.jl")
include("fw-as-2d.jl")

function floydwarshall_bench(mtx_name, fn_list, truth_fn)
    println("------")
    println("Benchmark results for $mtx_name (time taken(s))")

    mtx = matrixdepot(mtx_name)
    
    A = redefault!(Tensor(SparseMatrixCSC{Float64}(mtx)), Inf)

    for fw_fn in fn_list
        fn_name = string(nameof(fw_fn)) 
        if fn_name == "fw_base_adv_finch_kernel"
            time = @belapsed fw_base_adv_finch_kernel($A)
        elseif fn_name == "fw_as_1d_finch_kernel"
            time = @belapsed fw_as_1d_finch_kernel($A)
        elseif fn_name == "fw_as_2d_finch_kernel"
            time = @belapsed fw_as_2d_finch_kernel($A)
        else
            throw("Unidentified function name!")
        end

        if fn_name == string(nameof(truth_fn))
            is_correct = true
        else
            is_correct = truth_fn(A) == fw_fn(A)
        end

        println("$(nameof(fw_fn)): $time [Correct: $(is_correct)]")
    end

    println("------")
end


floydwarshall_bench(
    "Pajek/California", 
    [
        fw_base_adv_finch_kernel,
        fw_as_1d_finch_kernel,
        fw_as_2d_finch_kernel
    ], 
    fw_base_finch_kernel
)

# floydwarshall_bench("Pajek/EVA")
# floydwarshall_bench("vanHeukelum/cage8")
# mtx = [   
#         0    1    5    Inf Inf Inf; 
#         Inf 0    3    12   Inf Inf; 
#         Inf Inf 0    Inf 2    Inf;
#         Inf Inf Inf 0    Inf Inf;
#         Inf Inf Inf Inf 0    2;
#         Inf Inf Inf 2    Inf 0;
#     ]
