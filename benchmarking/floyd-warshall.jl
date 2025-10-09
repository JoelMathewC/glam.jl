using Finch
using MatrixDepot
using BenchmarkTools
using SparseArrays

function floydwarshall_finch_kernel(edges)
    (n, m) = size(edges)
    @assert n == m

    dists_prev = Tensor(Dense(Dense(Element(Inf))), n, n)
    dists = Tensor(Dense(Dense(Element(Inf))), n, n)
    @finch begin
        for j = _ 
            for i = _
                dists_prev[i,j] = edges[i, j]
            end
        end
    end

    any_active = Scalar(false)

    for t in 1:n
        println("Iteration $t")
        @finch begin
            for j = _
                for i = _
                    dists[i,j] = dists_prev[i,j]
                end
            end
        end
        
        @finch begin
            for j = _
                for k = _
                    if dists_prev[k, j] < Inf
                        for i = _
                            if dists_prev[i, k] < Inf
                                let d = dists_prev[i,k] + dists_prev[k,j]
                                    dists[i,j] <<min>>= d
                                end
                            end
                        end
                    end
                end
            end
        end

        @finch begin
            any_active .= false
            for i = _
                for j = _
                    any_active[] |= (dists[j,i] != dists_prev[j,i])
                end
            end
        end
        if !any_active[]
            break
        end

        dists_prev, dists = dists, dists_prev
    end

    return dists
end

function floydwarshall_finch(mtx)
    A = redefault!(Tensor(SparseMatrixCSC{Float64}(mtx)), Inf)
    time = @belapsed floydwarshall_finch_kernel($A)
    output = floydwarshall_finch_kernel(A)
    return (; time = time, mem = Base.summarysize(A), output = output)
end

res = floydwarshall_finch(matrixdepot("Pajek/EVA"))
print(res.time)