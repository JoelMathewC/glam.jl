using Finch

function floyd_warshall_einsum(adj_matrix)
    G = Tensor(Dense(SparseList(Element(10^8))), adj_matrix)
    (n, _) = size(G)

    for t in 1:n
        @einsum G[i,j] <<min>>= G[i,k] + G[k,j]
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
        10^8 0    10^8; 
        10^8 10^8 0
    ]

print(floyd_warshall_einsum(adj_matrix))

# Negative Cycle case
# adj_matrix = [   
#         0    10^8  1; 
#         1    0     10^8; 
#         10^8 -3    0
#     ]

# adj_matrix = [   
#         0    1    5    10^8 10^8 10^8; 
#         10^8 0    3    12   10^8 10^8; 
#         10^8 10^8 0    10^8 2    10^8;
#         10^8 10^8 10^8 0    10^8 10^8;
#         10^8 10^8 10^8 10^8 0    2;
#         10^8 10^8 10^8 2    10^8 0;
#     ]