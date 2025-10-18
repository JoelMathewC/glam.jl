using Finch
using MatrixDepot
using BenchmarkTools
using SparseArrays

function fw_as_2d_finch_kernel(edges)
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

    active = Tensor(Dense(SparseByteMap(Pattern())), n, n)
    active_next = Tensor(Dense(SparseByteMap(Pattern())), n, n)
    any_active = Scalar(false)

    @finch begin
        for j = _
            for i = _
                active[i,j] = edges[i,j] < Inf
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
        
        @finch mode=:fast begin
            active_next .= 0
            for j = _
                for k = _
                    for i = _
                        if active[k,j] || active[i,k]
                            let d_ij = dists_prev[i,k] + dists_prev[k,j]
                                dists[i,j] <<min>>= d_ij
                                active_next[i,j] |= d_ij < dists_prev[i,j]
                            end
                        end
                    end
                end
            end
        end

        @finch begin
            any_active .= false
            for j = _
                for i = _
                    any_active[] |= active_next[i,j]
                end
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