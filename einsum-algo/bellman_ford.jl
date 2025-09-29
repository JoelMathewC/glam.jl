using Finch

function bellman_ford_einsum(adj_matrix,src)
    G = Tensor(Dense(SparseList(Element(Inf))), adj_matrix)
    (n, _) = size(G)

    D = Tensor(Dense(Element(Inf)), n)
    @finch D[src] = 0

    for t in 1:n
        @einsum D[i] <<min>>= D[j] + G[j,i]
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