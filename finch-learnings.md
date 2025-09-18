# Finch Learnings

1. `tensor_tree(...)` - Wrapping any tensor within this will allow you to inspect the storage
2. `Tensor(Dense(SparseList(Element(1.0))),[0 1 1; 0 0 0; 0 0 0])` - The Element indicator here specifies the fill value
3. The matrices in Finch are stored in column major form
4. `tensor_tree(Tensor(Dense(Element(1.0)),[0; 1; 1]))` - this works
5. `tensor_tree(Tensor(Dense(Element(1.0)),[0 1 1]))` - But this does not because of dimension mismatch
6. Some gotchas
   1. Assigning the empty element (the value in `Element(*)`) to a tensor causes no finch code to be generated. Additionally, assigning non-empty elements leads to all elements becoming empty and error being thrown. Seems like this is because, empty elements can be converted to non-sparse once and that's it. They can not be converted back to empty elements or other non-empty elements.
        ```
        julia> pending = Tensor(Dense(SparseList(Element(0))), [1 1 1])
            1  1  1

        julia> @finch pending[1,1] = 0
            NamedTuple()

        julia> pending
            1  1  1

        julia> @finch_code pending[1,1] = 0 
            :(())

        julia> pending = Tensor(Dense(SparseList(Element(0))), [1 1 1])
            1  1  1
        
        julia> @finch pending[1,src] = 2
            ERROR: Finch.FinchProtocolError("SparseListLevels cannot be updated multiple times")
        ```