# An implementation of Floyd Warshall with two modifications
#   1. Skip processing of edges (k,i) that have infinite distances between each other
#   2. Early stop based on whether there is a diff between dists_prev and dists

using Finch
using MatrixDepot
using BenchmarkTools
using SparseArrays

function fw_base_adv_finch_kernel(edges)
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