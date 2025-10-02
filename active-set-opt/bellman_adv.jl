# Compare against the one in the finch repo when done
using Finch

function bellman_ford_einsum(adj_matrix,src)
    G = Tensor(Dense(SparseList(Element(Inf))), adj_matrix)
    (n, _) = size(G)

    D_prev = Tensor(Dense(Element(Inf)), n)
    @finch D_prev[src] = 0
    D = Tensor(Dense(Element(Inf)), n)

    active_prev = Tensor(SparseByteMap(Pattern()), n)
    active_prev[src] = true
    active = Tensor(SparseByteMap(Pattern()), n)
    any_active = Scalar(false)

    for t in 1:n
        @finch for i=_; if active_prev[i] D[i] <<min>>= D_prev[i] end end

        @finch begin
            active .= false
            for i = _
                if active_prev[i]
                    for j = _
                        let d = D_prev[i] + G[i, j]
                            D[j] <<min>>= d
                            active[j] |= d < D_prev[i]
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
        0    1    1; 
        Inf 0    Inf; 
        Inf Inf 0
    ]

print(bellman_ford_einsum(adj_matrix,1))

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