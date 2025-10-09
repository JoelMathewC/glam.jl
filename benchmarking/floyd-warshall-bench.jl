using Finch
using MatrixDepot
using BenchmarkTools
using SparseArrays

# Transform from sparse to dense


function floydwarshall_activeset_kernel(edges)
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
    active_count =  Scalar(0)

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
                    if (active[j] || active[k])
                        for i = _
                                let d = dists_prev[i,k] + dists_prev[k,j]
                                    dists[i,j] <<min>>= d
                                    active_next[j] |= d < dists_prev[i,j]
                                end
                        end
                    end
                end
            end
        end

        @finch begin
            any_active .= false
            a
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

function floydwarshall_earlystop_kernel(edges)
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
                    for i = _
                        let d = dists_prev[i,k] + dists_prev[k,j]
                            dists[i,j] <<min>>= d
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

function floydwarshall_2dactiveset(edges)
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
    process_count = Scalar(0)

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
                        if active[k,j] #|| active[i,k]
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

        # @finch begin
        #     process_count .= 0
        #     for j = _
        #         for k = _
        #             process_count[] += active[k,j]
        #         end
        #     end
        # end

        dists_prev, dists = dists, dists_prev
        active, active_next = active_next, active
    end

    return dists
end

function floydwarshall_kernel(edges)
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

        dists_prev, dists = dists, dists_prev
    end

    return dists
end

function floydwarshall_bench(mtx_name)
    println("Benchmark results for $mtx_name")
    mtx = matrixdepot(mtx_name)
    A = redefault!(Tensor(SparseMatrixCSC{Float64}(mtx)), Inf)
    # time_naive = @belapsed floydwarshall_kernel($A)
    time_earlystop = @belapsed floydwarshall_earlystop_kernel($A)
    # time_activeset = @belapsed floydwarshall_2dactiveset($A)
    time_activeset = @belapsed floydwarshall_activeset_kernel($A)

    # println("Naive (time taken(s)): $time_naive")
    println("Early Stop (time taken(s)): $time_earlystop")
    println("Active Set (time taken(s)): $time_activeset")

    output_naive = floydwarshall_earlystop_kernel(A)
    output_activeset = floydwarshall_2dactiveset(A)

    is_different = Scalar(false)
    @finch begin
        for i = _
            for j = _
               is_different[] |= (output_naive[j,i] != output_activeset[j,i])
            end
        end
    end

    println("Does output from active set version match naive: $(!is_different.val)")
    println("------")
end

# floydwarshall_bench("Pajek/EVA")
floydwarshall_bench("Pajek/California")
# floydwarshall_bench("vanHeukelum/cage8")
