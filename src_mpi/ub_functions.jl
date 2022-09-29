module ub_functions

using Distances
using lb_functions, opt_functions
using TimerOutputs: @timeit, get_timer
export getUpperBound, getRootUpperBound

# get the upper bound for the root nodes
function getRootUpperBound(X, k)
    node_UB, group_centers_index = init_ub_heuristic(X, k)
    node_centers = X[:, group_centers_index]
    node_UB, node_assign = obj_assign(node_centers, X)

    return node_UB, node_assign, group_centers_index, node_centers
end

# get the upper bound for child nodes
function getUpperBound(X, k, t_assign, group_centers)
    group_centers_index = UB_improve(X, t_assign, k, group_centers)
    node_centers = center_index_coor(X, group_centers_index) # k
    node_UB, node_assign = obj_assign(node_centers, X) # opt_functions

    return node_UB, node_assign, group_centers_index, node_centers
end

# end of the module
end