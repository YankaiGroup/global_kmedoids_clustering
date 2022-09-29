module branch

using Nodes
using TimerOutputs: @timeit, get_timer
using MPI, Statistics

export branch!, CheckSinglePoint, CheckExistence, branch_median!, CheckAllPoints

function SelectVarMaxRange(node::Node)
    dif = node.upper -node.lower
    ind = findmax(dif)[2]
    return ind[1], ind[2]
end

function SelectVardMaxLBCenterRange(group_centers::Array{Float64})
    d = size(group_centers)[1]
    k = size(group_centers)[2]
    dif = zeros(Float64, d, k)

    for clst in 1:k::Int
        for dim in 1:d::Int
            dif[dim, clst] = maximum(group_centers[dim, clst,:]) - minimum(group_centers[dim, clst,:])
        end
    end
    #println("LBCenterRange:   ",dif)
    ind = findmax(dif)[2]
    return ind[1], ind[2]
end

# check point existence in a region
# early stop if there is at least one point in each kernel region
function CheckExistence(X::Matrix{Float64}, lower::Matrix{Float64}, upper::Matrix{Float64}, k::Int) 
    ~, n = size(X)
    flag_find = falses(k)
    for i = 1:k::Int
       #if fixed_centers[i] == 0
            tmp_lower = lower[:, i]
            tmp_upper = upper[:, i]
            for j = 1:n::Int
                if CheckSinglePoint(X[:, j], tmp_lower, tmp_upper)
                    flag_find[i] = true
                    break
                end
            end
        #end
    end

    return sum(flag_find) == k ? true : false
end

# check if a point is in lower-upper restriction of one kernel
function CheckSinglePoint(point::Vector{Float64}, lower::Array{Float64}, upper::Array{Float64})
    point = vec(point)
    lower = vec(lower)
    upper = vec(upper)
    d = length(point)
    for i = 1:d::Int
        if point[i] < lower[i] || point[i] > upper[i]
            return false
        end
    end
    return true
end

# check only one point in a kernel region
# early stop if there is two points in a region
function CheckTheOnlyPoint(X::Matrix{Float64}, lower::Matrix{Float64}, upper::Matrix{Float64}, k::Int, fixed_centers::Vector{Int}) 
    ~, n = size(X)
    tmp_fixed_centers = zeros(Int, k)
    for i = 1:k::Int
        num_find = 0
        coor_find = 0
        tmp_lower = lower[:, i]
        tmp_upper = upper[:, i]
        if fixed_centers[i] != 0 # fixed center in ith region 
            if CheckSinglePoint(X[:, Int(fixed_centers[i])], tmp_lower, tmp_upper)
                num_find += 1
                coor_find = Int(fixed_centers[i])
            end
        else # no fixed center
            for j = 1:n::Int
                if CheckSinglePoint(X[:, j], tmp_lower, tmp_upper)
                    num_find += 1
                    if num_find > 1
                        break
                    end
                    coor_find = j
                end
            end
        end
        if num_find == 1
            tmp_fixed_centers[i] = coor_find
        end
    end

    return tmp_fixed_centers
end

function CheckAllPoints(X::Matrix{Float64}, lower::Matrix{Float64}, upper::Matrix{Float64})
    d, n = size(X)
    d, k = size(lower)
    n_lower = zeros(Float64, d, k)
    n_upper = zeros(Float64, d, k)
    all_num = zeros(Int, k)
    kernel_index = Vector{Vector{Int}}(undef, k)
    for j = 1:k::Int
        tmp_kernel_index = Int[]
        for i = 1:n::Int
            if sum(view(X,:,i) .>= view(lower,:,j))==d && sum(view(X,:,i) .<= view(upper,:,j))==d
                push!(tmp_kernel_index, i)
                all_num[j] +=1
                if all_num[j] == 1
                    n_lower[:, j] = X[:, i]
                    n_upper[:, j] = X[:, i]
                elseif all_num[j] >1
                    for l in 1:d::Int
                        if n_lower[l, j] > X[l, i]
                            n_lower[l, j] = X[l, i]
                        end
                        if n_upper[l, j] < X[l,i]
                            n_upper[l, j] = X[l, i]
                        end
                    end
                end
            end
        end
        kernel_index[j] = tmp_kernel_index
    end

    return n_lower, n_upper, kernel_index
end

function branch_one_node!(X::Matrix{Float64}, nodeList::Vector{Node}, LB_List::Vector{Float64}, bVarIdx::Int, bVarIdy::Int, bValue::Float64, node::Node, node_LB::Float64, k::Int, insert_id::Int, right::Bool=true)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0

    d, n = size(X)
    del_num = 0
    lower = copy(node.lower)
    upper = copy(node.upper)
    if right # right branch
        lower[bVarIdx, bVarIdy] = bValue
        for j = 2:k::Int # symmetric breaking
            if lower[1, j] <= lower[1, j-1]
                lower[1, j] = lower[1, j-1]
            end
        end
        output = "right"
    else # left branch
        upper[bVarIdx, bVarIdy] = bValue
        for j = 1:(k-1)::Int  # symmetric breaking
            if upper[1, k-j] >= upper[1, k-j+1]
                upper[1, k-j] = upper[1, k-j+1]
            end
        end
        output = "left"
    end

    dulp_flag = false
    exist_flag = false
    if sum(lower .<= upper) == d * k
        exist_flag = CheckExistence(X, lower, upper, k)
        if exist_flag == true # check point existence in this node
            fixed_centers = CheckTheOnlyPoint(X, lower, upper, k, node.fixed_centers)
            for i in unique(fixed_centers) # in case of same fixed_centers
                if i != 0
                    if count(==(i), fixed_centers) > 1
                        dulp_flag = true
                    end
                end
            end
            # if rank == 0
            #     println(output, " - exist_flag:", exist_flag, ", dulp_flag:", dulp_flag)
            # end
            if dulp_flag != true # check duplicated fixed_centers
                lower, upper, ~ = CheckAllPoints(X, lower, upper)
                right_node = Node(lower, upper, node.level + 1, node_LB, node.lambda, node.group_centers, fixed_centers, node.assign_idx, node.best_lambda, node.best_alpha, node.best_alpha_index_k, node.fbbt_type, [bVarIdx, bVarIdy], node.CF_LB)
                insert!(nodeList, insert_id, right_node)
                insert!(LB_List, insert_id, node_LB)
            else # delete a right node for duplicated fixed_centers
                del_num += 1
                if rank == 0
                    println("************** delete a $output node for duplicated fixed_centers **************")
                end
            end
        else # delete a right node for existence
            del_num += 1
            if rank == 0
                println("************** delete a $output node for existence **************")
            end
        end
    end

    return del_num
end

function branch!(X::Matrix{Float64}, nodeList::Vector{Node}, LB_List::Vector{Float64}, bVarIdx::Int, bVarIdy::Int, bValue::Float64, node::Node, node_LB::Float64, k::Int, check::Bool, method::String)
    # check the specific nodeid to insert
    insert_id = searchsortedfirst(LB_List, node_LB)
    del_num = 0

    # right node
    if check == true
        del_num += branch_one_node!(X, nodeList, LB_List, bVarIdx, bVarIdy, bValue, node, node_LB, k, insert_id, true)
    end
    
    # left node
    if check == true
        del_num += branch_one_node!(X, nodeList, LB_List, bVarIdx, bVarIdy, bValue, node, node_LB, k, insert_id, false)
    end

    return del_num
end

end