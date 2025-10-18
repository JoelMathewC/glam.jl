using Finch

function floyd_warshall_adv_einsum(adj_matrix)
    (n, _) = size(adj_matrix)

    # We need two lists here since the computation of D at time t depends on D at time t-1
    D_prev = Tensor(Dense(SparseList(Element(Inf))), adj_matrix)
    D = Tensor(Dense(SparseList(Element(Inf))), n, n)

    # We need two lists here since the computation of active at time t+1 depends on active at time t
    active = Tensor(SparseByteMap(Pattern()), n)
    active_next = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    # Start with all vertices being active_next
    @finch for i = _ 
        active[i] = true
    end

    for t in 1:n
        # TODO: This will take O(N^2), improve using active_next set
        @einsum D[i,j] = D_prev[i,j]
        
        @finch begin
            active_next .= 0
            for j = _
                for k = _
                    if active[j] || active[k]
                        for i = _
                            let d = D_prev[i,k] + D_prev[k,j]
                                D[i,j] <<min>>= d
                                active_next[j] |= d < D_prev[i,j]
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

        D_prev, D = D, D_prev
        active, active_next = active_next, active
    end

    return D
end

adj_matrix = [   
        0    1    5    Inf Inf Inf; 
        Inf 0    3    12   Inf Inf; 
        Inf Inf 0    Inf 2    Inf;
        Inf Inf Inf 0    Inf Inf;
        Inf Inf Inf Inf 0    2;
        Inf Inf Inf 2    Inf 0;
    ]

result = floyd_warshall_adv_einsum(adj_matrix)

for i in 1:size(adj_matrix,1)
    for j in 1:size(adj_matrix,1)
        print("$(result[i,j]) ")
    end
    print("\n")
end

# Negative Cycle case
# adj_matrix = [   
#         0    Inf  1; 
#         1    0     Inf; 
#         Inf -3    0
#     ]

# adj_matrix = [   
#         0    1    5    Inf Inf Inf; 
#         Inf 0    3    12   Inf Inf; 
#         Inf Inf 0    Inf 2    Inf;
#         Inf Inf Inf 0    Inf Inf;
#         Inf Inf Inf Inf 0    2;
#         Inf Inf Inf 2    Inf 0;
#     ]