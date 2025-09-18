using Finch

function prims_einsum(adj_matrix,src)
    G = Tensor(Dense(SparseList(Element(10^8))), adj_matrix)
    (n, _) = size(G)

    # Initialization for mst
    mst = Tensor(Dense(SparseList(Element(0))), n, n)
    @finch mst[src,src] = 0

    # Initializations to calculate pending of each node
    pending = Tensor(Dense(SparseList(Element(0))), 1, n)

    # Declare the starting state of the frontier
    frontier = Tensor(Dense(SparseList(Element(0))), 1, n)
    @finch frontier[1,src] = 1

    fnz_count = Scalar(1)

    # Loop till the frontier is no longer empty
    for t in 1:n
        @einsum c[] <<minby>>= frontier[k,i] * G[i,j] * pending[k,j] => (i,j)
        @finch mst[c[][0],c[][1]] = 1
        @finch pending[1,c[][1]] = 10^8

        frontier = Tensor(Dense(SparseList(Element(0))), 1, n)
        @finch frontier[1,c[][1]] = 1
    end

    return mst
end

adj_matrix = [   
        0    1    1; 
        0    0    0; 
        0    1    0
    ]

print(prims_einsum(adj_matrix,1))