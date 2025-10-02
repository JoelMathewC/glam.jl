using Finch

function floyd_warshall_adv_einsum(adj_matrix)
    (n, _) = size(adj_matrix)

    D_prev = Tensor(Dense(SparseList(Element(Inf))), adj_matrix)
    D = Tensor(Dense(SparseList(Element(Inf))), n, n)

    active_prev = Tensor(SparseByteMap(Pattern()), n)
    @einsum active_prev[i] = active_prev[i] | true
    active = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    for t in 1:n
        @einsum D[i,j] = D_prev[i,j]
        
        active = Tensor(SparseByteMap(Pattern()), n)
        @finch begin
            for j = _
                for i = _
                    if active_prev[i] || active_prev[j]
                        for k = _
                            let d = D_prev[i,k] + D_prev[k,j]
                                D[i,j] <<min>>= d
                                active[i] |= d < D_prev[i,j]
                                active[j] |= d < D_prev[i,j]
                            end
                        end
                    end
                end
            end
        end

        @finch begin
            any_active .= false
            for i = _
                any_active[] |= active[i]
            end
        end
        if !any_active[]
            break
        end

        D_prev, D = D, D_prev
        active_prev, active = active, active_prev
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