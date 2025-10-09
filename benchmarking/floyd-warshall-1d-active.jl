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

    active = Tensor(SparseByteMap(Pattern()), n)
    active_next = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    process_count = Scalar(0)

    @finch for i = _ 
        active[i] = true
    end

    for t in 1:n
        println("Starting iteration=$t")
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
        println("Completed FW step for t=$t")

        @finch begin
            any_active .= false
            for i = _
                any_active[] |= active_next[i]
            end
        end
        if !any_active[]
            break
        end

        @finch begin
            process_count .= 0
            for j = _
                for k = _
                    process_count[] += (dists_prev[k, j] < Inf) && (active[j] || active[k])
                end
            end
        end

        println("The active nnz count is $process_count")

        dists_prev, dists = dists, dists_prev
        active, active_next = active_next, active
    end

    return dists
end

function floydwarshall_finch(mtx)
    A = redefault!(Tensor(SparseMatrixCSC{Float64}(mtx)), Inf)
    time = @belapsed floydwarshall_finch_kernel($A)
    output = floydwarshall_finch_kernel(A)
    return (; time = time, mem = Base.summarysize(A), output = output)
end


res = floydwarshall_finch(matrixdepot("Gleich/wb-cs-stanford"))
print(res.time)