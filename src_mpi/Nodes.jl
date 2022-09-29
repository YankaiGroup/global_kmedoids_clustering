module Nodes

using Printf
using SparseArrays

export Node, printNodeList

struct Node
    lower::Union{Nothing, Matrix{Float64}} # d * k
    upper::Union{Nothing, Matrix{Float64}} # d * k
    level::Int
    LB::Float64
    lambda::Union{Nothing, Vector{Float64}} # n, lambda in LD
    group_centers::Union{Nothing, Vector{Int}} # k, indices of cluster centers 
    fixed_centers::Union{Nothing, Vector{Int}}  # k, fixed center indices by one point, 0: not fixed
    assign_idx::Union{Nothing, SparseVector{Int64, Int64}} # n, checked assigned samples to cluster k
    best_lambda::Union{Nothing, Vector{Float64}} # n, lambda with best LB, used in fbbt_tighter
    best_alpha::Union{Nothing, Vector{Float64}} # n, lambda with best LB, used in fbbt_tighter
    best_alpha_index_k::Union{Nothing, Vector{Int}} # k, alpha index with best LB, used in fbbt_tighter
    fbbt_type::Bool # boolean, true - fbbt_tighter (LD as LB), false - fbbt (TD as LB)
    bVarIds::Union{Nothing, Vector{Int}} # [d, k], branched variable and cluster indices
    CF_LB::Float64
end

Node() = Node(nothing, nothing, -1, -1e15, nothing, nothing, nothing, nothing, nothing, nothing, nothing, false, [0, 0], -1e15)


# function to print the node in a neat form
function printNodeList(nodeList::Vector{Node})
    for i in 1:length(nodeList)::Int
        println("******** Node", i, " ********")
        # println("Lower: ", map(x -> @sprintf("%.1f",x), getfield(nodeList[i],:lower))) # reserve 3 decimal precision
        # println("Upper: ", map(x -> @sprintf("%.1f",x), getfield(nodeList[i],:upper)))
        println("Level: ", getfield(nodeList[i],:level)) # integer
        println("LB: ", map(x -> @sprintf("%.3f",x), getfield(nodeList[i],:LB)))
        println("******** end ********")
    end
end


end