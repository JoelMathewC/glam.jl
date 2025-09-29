using Finch

function floyd_warshall_einsum(adj_matrix,src)
    D_prev = Tensor(Dense(SparseList(Element(Inf))), adj_matrix)
    (n, _) = size(D_prev)
    D = Tensor(Dense(SparseList(Element(Inf))), n, n)

    active_prev = Tensor(SparseByteMap(Pattern()), n)
    @finch active_prev[src] = true
    active = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    for t in 1:n
        @finch begin
            active .= false
            for i = _
                for k = _
                    if active_prev[i] || active_prev[k]
                        for j = _
                            let d = D_prev[i,k] + edges[k, j]
                                D[i,j] <<min>>= d
                                active[i,j] |= d < D_prev[i,j]
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
        dists_prev, dists = dists, dists_prev
        active_prev, active = active, active_prev
    end

    for t in 1:n
        if G[t,t] < 0
            throw("Negative cycle exists!")
        end
    end

    return G
end

adj_matrix = [   
        0    1    1; 
        Inf 0    Inf; 
        Inf Inf 0
    ]

print(floyd_warshall_einsum(adj_matrix,1))

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