# Base implementation of Floyd Warshall with two modifications

using Finch
using MatrixDepot
using BenchmarkTools
using SparseArrays

function fw_base_finch_kernel(edges)
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
                    for i = _
                        let d = dists_prev[i,k] + dists_prev[k,j]
                            dists[i,j] <<min>>= d
                        end
                    end
                end
            end
        end

        dists_prev, dists = dists, dists_prev
    end

    return dists
end