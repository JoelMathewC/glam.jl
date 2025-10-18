using Finch
using MatrixDepot
using BenchmarkTools
using SparseArrays

function fw_as_1d_finch_kernel(edges)
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

    active = Tensor(SparseByteMap(Pattern()), n)
    active_next = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    @finch for i = _ 
        active[i] = true
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
            active_next .= 0
            for j = _
                for k = _
                    if (dists_prev[k, j] < Inf) && (active[j] || active[k])
                        for i = _
                            if dists_prev[i, k] < Inf
                                let d = dists_prev[i,k] + dists_prev[k,j]
                                    dists[i,j] <<min>>= d
                                    active_next[j] |= d < dists_prev[i,j]
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
                any_active[] |= active_next[i]
            end
        end
        if !any_active[]
            break
        end

        dists_prev, dists = dists, dists_prev
        active, active_next = active_next, active
    end

    return dists
end