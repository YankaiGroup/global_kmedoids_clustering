module bb_functions

using MPI
using Printf
using Random
using Statistics, Distances
using TimerOutputs: @timeit, get_timer
using Clustering
using SparseArrays
using lb_functions, ub_functions, opt_functions, branch, Nodes, fbbt, probing

export branch_bound, grouping

maxiter = 3000000
max_trial = 30
tol = 1.0e-6
mingap = 1.0e-3
time_lapse = 2*60 # 60s*time_lapse

# function to record the finish time point
time_finish(seconds) = round(Int, 10^9 * seconds + time_ns())

# assign data to each core
function grouping(n::Int, ncores::Int)
    groups = Vector{Vector{Int}}(undef, ncores)
    group_num = div(n, ncores) # 1046910 // 4000 = 261; 3999 * 261 = 1043739; 3171
    # 261 * x + 262 * y = 1046910; x + y = 4000; x = 262*4000 - 1046910; y = 4000 - x
    left_num = (group_num+1) * ncores - n
    for i = 1:left_num::Int # distribute kernel samples to be calculated in each core
        group = Array{Int}((i-1)*group_num+1 : i*group_num) 
        groups[i] = group
    end
    for i = (left_num+1)::Int : ncores::Int
        pure_i = i - left_num
        pure_gp_num = group_num + 1
        if i != ncores
            group = Array{Int}(left_num*group_num + (pure_i-1) * pure_gp_num + 1 : left_num*group_num + pure_i * pure_gp_num)
        else
            group =  Array{Int}(left_num*group_num + (pure_i-1) * pure_gp_num + 1 : n)
        end
        groups[i] = group
    end
    # for i = 1:ncores
    #     println(i, " - ", size(groups[i]))
    # end
    return groups, group_num
end

# during iteration: 
# LB = node.LB is the smallest LB among all node and is in current iteration
# UB is the best UB, and node_UB the updated UB(it is possible: node_UB > UB) of current iteration (after run getUpperBound)
# node_LB is the updated LB of current iteration (after run probing or getLowerBound_adptGp_LD)
function branch_bound(X::Matrix{Float64}, dmat::Matrix{Float64}, groups::Vector{Vector{Int}}, k::Int, method::String = "BB+Basic", mode::String = "fixed", solver::String = "CPLEX", check::Bool = true, lambda_updater::String = "periodical", fbbt_method::String = "fbbt", LD_trial::Int = 30, probing_flag::Bool = true, gc_flag::Bool = true)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    world_size = MPI.Comm_size(comm)
    p_root = 0
    counts = [length(groups[i]) for i in 1:world_size::Int]

    if rank == p_root
        print("total processors: $world_size\n")
        d, n = size(X);
        lower, upper = opt_functions.init_bound(X, d, k)
    else
        d = nothing
        n = nothing
        lower = nothing
        upper = nothing
    end
    MPI.Barrier(comm)
    d = MPI.bcast(d, p_root, comm)
    n = MPI.bcast(n, p_root, comm)
    lower = MPI.bcast(lower, p_root, comm)
    upper = MPI.bcast(upper, p_root, comm)
    MPI.Barrier(comm)

    UB = Inf # best upper bound
    LB = -Inf # \beta_l
    CF_LB = -Inf
    node_LB = LB
    node_UB = UB
    max_LB = Inf # used to save the best lower bound when all node solved (smallest but within the mingap)

    r_start_time = time_ns()
    
    ### root node ###
    centers = nothing;
    group_centers = nothing;
    fixed_centers = zeros(Int, k);
    root = Node(lower, upper, 0, LB, nothing, nothing, fixed_centers, spzeros(Int, n), nothing, nothing, nothing, false, [0, 0], CF_LB); 
    @timeit get_timer("Shared") "iter 1 - heuristic UB" begin
    ## init UB by heuristic search ##
    if rank == p_root
        node_UB, node_assign, group_centers_index, node_centers = getRootUpperBound(X, k)
        # node_UB, group_centers_index = init_ub_heuristic(X, k)
        # node_centers = X[:, group_centers_index]
        # node_UB, node_assign = obj_assign(node_centers, X)
    else
        node_UB = nothing
        group_centers_index = nothing
        node_centers = nothing
        node_assign = nothing
    end
    MPI.Barrier(comm)
    node_UB = MPI.bcast(node_UB, p_root, comm)
    group_centers_index = MPI.bcast(group_centers_index, p_root, comm)
    node_centers = MPI.bcast(node_centers, p_root, comm)
    node_assign = MPI.bcast(node_assign, p_root, comm)
    MPI.Barrier(comm)
    GC.gc()

    clst_rlt = nothing
    if rank == p_root
        print("UB from kmeans: $node_UB\n")
    end

    lambda = nothing
    best_lambda = nothing
    best_alpha = nothing
    if (method == "BB+LD") # init lambda by local optimal
        lambda_lcl_2 = init_analytic_LD_lambda_2(dmat, groups[rank+1], group_centers_index, node_assign)
        lambda = MPI.Allgatherv!(lambda_lcl_2, VBuffer(similar(lambda_lcl_2, sum(counts)), counts), comm)
    end

    lambda_group_centers_index = group_centers_index #  # variable for lambda update after each node
    lambda_node_assign = node_assign # variable for lambda update after each node

    if rank == p_root
        centers = node_centers
    end
    node_UB = MPI.bcast(node_UB, p_root, comm)

    if lambda !== nothing # init lambda by local optimal
        root = Node(lower, upper, root.level, root.LB, lambda, group_centers_index, root.fixed_centers, root.assign_idx, root.best_lambda, root.best_alpha, root.best_alpha_index_k, root.fbbt_type, root.bVarIds, root.CF_LB);
    else
        root = Node(lower, upper, root.level, root.LB, root.lambda, group_centers_index, root.fixed_centers, root.assign_idx, root.best_lambda, root.best_alpha, root.best_alpha_index_k, root.fbbt_type, root.bVarIds, root.CF_LB);
    end
    if node_UB < UB
        UB = node_UB
    end    
    end # end for begin

    ### create nodelist ###
    iter = 0
    if rank == p_root
        nodeList = Node[]
        LB_List = Float64[];
        push!(nodeList, root)
        push!(LB_List, -1.0e15)
        println(" iter ", " left ", " lev  ", "       LB       ", "       UB      ", "      gap   ")
        calcInfo = [] # initial space to save calcuation information
    end
    
    # get program end time point
    end_time = time_finish(time_lapse*60) 
    overall_end_time = end_time

    ### inside main loop ###
    del_nums = 0
    kernel_delta = zeros(Int, 2)

    while true
        # equal to while nodeList != []
        nodeList_flag = false
        if rank == p_root
            if nodeList == []
                nodeList_flag = true
            end
        end
        MPI.Barrier(comm)
        nodeList_flag = MPI.bcast(nodeList_flag, p_root, comm)
        MPI.Barrier(comm)
        if nodeList_flag # empty nodeList
            break
        end

        # we start at the branch(node) with lowest Lower bound
        @timeit get_timer("Shared") "BB part one" begin
        if rank == p_root
            nodeid = 1
            node = nodeList[nodeid] # lowest LB node
            LB = node.LB
            deleteat!(nodeList, nodeid)
            deleteat!(LB_List, nodeid)
            # so currently, the global lower bound corresponding to node, LB = node.LB, groups = node.groups
            @printf "%-6d %-6d %-10d %-10.8f %-10.8f %-10.8f %s \n" iter length(nodeList) node.level LB UB (UB-LB)/min(abs(LB), abs(UB))*100 "%"
            # save calcuation information for result demostration
            push!(calcInfo, [iter, length(nodeList), node.level, LB, UB, (UB-LB)/min(abs(LB), abs(UB))])
        else
            node = nothing
        end
        MPI.Barrier(comm)
        node = MPI.bcast(node, p_root, comm)
        MPI.Barrier(comm)
        
        # time stamp should be checked after the retrival of the results
        if (iter == maxiter) || (time_ns() >= end_time)
            break
        end
        iter += 1
        end # end for begin
        ###### probing ######
        if probing_flag && node.level <= 8 && iter <= 100
            if rank == p_root
                @timeit get_timer("Shared") "probing" lwr, upr, node_LB = probing_base(X, k, node.lower, node.upper, node.LB, UB, mingap, node.bVarIds)
            else
                lwr = nothing
                upr = nothing
                node_LB = nothing
            end
            MPI.Barrier(comm)
            lwr = MPI.bcast(lwr, p_root, comm)
            upr = MPI.bcast(upr, p_root, comm)
            node_LB = MPI.bcast(node_LB, p_root, comm)
            MPI.Barrier(comm)

            node = Node(lwr, upr, node.level, node_LB, node.lambda, node.group_centers, node.fixed_centers, node.assign_idx, node.best_lambda, node.best_alpha, node.best_alpha_index_k, node.fbbt_type, node.bVarIds, node.CF_LB);
        end
        
        ###### Assignmen check ######
        if rank == p_root
            if method == "BB+LD"
                ass_chk = true
            else
                ass_chk = false
            end
            tmp_assign_idx = spzeros(Int, n)
            if ass_chk
                tmp_assign_idx = assign_check(X, k, node.lower, node.upper, node.assign_idx) 
                if rank == p_root
                    println("assign_idx_sum: ", sum(tmp_assign_idx))
                end
            end
        else
            tmp_assign_idx = nothing 
        end
        MPI.Barrier(comm)
        tmp_assign_idx = MPI.bcast(tmp_assign_idx, p_root, comm)
        MPI.Barrier(comm)
        node = Node(node.lower, node.upper, node.level, node.LB, node.lambda, node.group_centers, node.fixed_centers, copy(tmp_assign_idx), node.best_lambda, node.best_alpha, node.best_alpha_index_k, node.fbbt_type, node.bVarIds, node.CF_LB);

        ############# iteratively getLowerBound #######################
        node_LB = LB

        @timeit get_timer("Shared") "getLowerBound" begin
        if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            if rank == p_root
                println("analytic LB:",node_LB, " \t >= UB:", UB)
            end
        else
            if (method == "BB+LD") # BB+LD
                node_LB, node_UB, lambda, best_lambda, group_centers_index, best_alpha_index_k, fbbt_type, tmp_kernel_delta, best_alpha, CF_LB = lb_functions.getLowerBound_analytic_LD_subgradient(X, dmat, k, overall_end_time, groups, counts, iter, node, UB, lambda_updater, LD_trial, fbbt_method)
                node = Node(node.lower, node.upper, node.level, node.LB, lambda, group_centers_index, node.fixed_centers, node.assign_idx, best_lambda, best_alpha, best_alpha_index_k, fbbt_type, node.bVarIds, CF_LB);
                kernel_delta += tmp_kernel_delta
                if node_LB > 1.4*node_UB # in case of zero medoids in BB+LD
                    continue
                end
            elseif (method == "LD") # pure LD
                if rank == p_root
                    kernel_index = [Array(1:n), Array(1:n), Array(1:n)]
                    @timeit get_timer("Shared") "all_subgrad" node_LB, node_UB, lambda, best_lambda, medoids_index, best_alpha_index_k, best_alpha = analytic_LD_subgrad(X, k, overall_end_time, kernel_index, groups, counts, iter, UB, node.lambda, node.best_lambda, node.best_alpha, node.best_alpha_index_k, node.LB, node.group_centers, dmat, lambda_updater, LD_trial, 10000)
                    if node_UB < UB
                        UB = node_UB
                    end
                    if iter == 1 # store node_LB in the root node
                        max_LB = node_LB
                    end
                    push!(calcInfo, [iter, length(nodeList), 0, max_LB, UB, (UB-max_LB)/min(abs(max_LB), abs(UB))])
                    @printf "%-52d  %-10.4f %-10.4f %-7.4f %s \n" iter  max_LB UB (UB-max_LB)/min(abs(max_LB),abs(UB))*100 "%"
                    centers = X[:, medoids_index]
                    println("index: ", medoids_index)
                    println("centers: ",centers)

                    return centers, UB, calcInfo
                end
            elseif (method == "BB+Basic") # closedForm
                if rank == p_root
                    n_lower = node.lower
                    n_upper = node.upper
                    zero_sample_flag = false
                    @timeit get_timer("Shared") "getLowerBound_analytic" begin
                    if zero_sample_flag == false
                        node_LB, group_centers, group_centers_index = lb_functions.getLowerBound_analytic(X, k, n_lower, n_upper) # getLowerBound with closed-form expression
                    else
                        node_LB = UB*2 # prevent further branch 
                        group_centers_index = node.group_centers
                    end
                    node = Node(n_lower, n_upper, node.level, node.LB, node.lambda, group_centers_index, node.fixed_centers, node.assign_idx, node.best_lambda, node.best_alpha, node.best_alpha_index_k, node.fbbt_type,node.bVarIds, node.CF_LB) 
                    end # for begin
                end
            end

        end
        end # end for begin
        MPI.Barrier(comm)
        ### update UB and delete nodes accordingly ###
        delete_nodes = Int[]
    
        @timeit get_timer("Shared") "Update UB" begin
        if rank == p_root
            group_centers_coor = center_index_coor(X, node.group_centers) # k
            t_UB, t_assign = obj_assign(group_centers_coor, X) # opt_functions
            if method == "BB+LD"
                # new_group_centers = UB_improve(X, t_assign, k, node.group_centers)
                # new_group_centers_coor = center_index_coor(X, new_group_centers) # k
                # new_t_UB, new_t_assign = obj_assign(new_group_centers_coor, X) # opt_functions
                new_t_UB, new_t_assign, new_group_centers, new_group_centers_coor = getUpperBound(X, k, t_assign, node.group_centers)
                if new_t_UB < t_UB
                    if rank==p_root
                        println("new_t_UB is smaller than t_UB: $new_t_UB < $t_UB")
                    end
                    t_UB = new_t_UB
                    t_assign = new_t_assign
                    group_centers_coor = new_group_centers_coor
                    node = Node(node.lower, node.upper, node.level, node.LB, node.lambda, new_group_centers, node.fixed_centers, node.assign_idx, node.best_lambda, node.best_alpha, node.best_alpha_index_k, node.fbbt_type, node.bVarIds, node.CF_LB)
                end
            end
            if (t_UB < UB)
                node_UB = t_UB	      
            else
                node_UB = UB	  
            end  
        else
            node_UB = nothing
            group_centers_coor = nothing
            t_assign = nothing
            node = nothing
        end
        MPI.Barrier(comm)
        node_UB = MPI.bcast(node_UB, p_root, comm)
        group_centers_coor = MPI.bcast(group_centers_coor, p_root, comm)
        t_assign = MPI.bcast(t_assign, p_root, comm)
        node = MPI.bcast(node, p_root, comm)
        MPI.Barrier(comm)
        end # end for begin  

        centers_update_flag = false
        @timeit get_timer("Shared") "delete LB > UB nodes" begin  
        # println("node_UB: ", node_UB, " UB:", UB, " node_UB <= UB:", (node_UB <= UB))
            if (node_UB < UB) # in case of unequal t_UB and UB in BB+LD
                UB = node_UB
                centers = group_centers_coor
                lambda_group_centers_index = node.group_centers
                lambda_node_assign = t_assign
                centers_update_flag = true
                # the following code delete branch with lb close to the global upper bound
                if rank == p_root
                    delete_nodes = Int[]
                    for (idx,n) in enumerate(nodeList)
                        if (((UB-n.LB)<= mingap) || ((UB-n.LB) <=mingap*min(abs(UB), abs(n.LB))))
                            push!(delete_nodes, idx)
                        end
                    end
                    deleteat!(nodeList, sort(delete_nodes))
                    deleteat!(LB_List, sort(delete_nodes))
                end
            end
        end # end for begin
        MPI.Barrier(comm)

        if method == "BB+LD" &&  centers_update_flag == true # init lambda by local optimal if centers are updated
            if rank == p_root
                println("centers updated!")
            end
            lambda_lcl_2 = init_analytic_LD_lambda_2(dmat, groups[rank+1], lambda_group_centers_index, lambda_node_assign)
            lambda = MPI.Allgatherv!(lambda_lcl_2, VBuffer(similar(lambda_lcl_2, sum(counts)), counts), comm)
            node = Node(node.lower, node.upper, node.level, node.LB, lambda, group_centers_index, node.fixed_centers, node.assign_idx, node.best_lambda, node.best_alpha, node.best_alpha_index_k, node.fbbt_type, node.bVarIds, node.CF_LB) 
        end

        @timeit get_timer("Shared") "branch" begin
        # here this condition include the condition UB < node_LB and the condition that current node's LB is close to UB within the mingap
        # Such node no need to branch
        del_num = 0
        if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
            # used to save the best lower bound when all node solved (smallest but within the mingap)
            if node_LB < max_LB
                max_LB = node_LB
                if rank == p_root
                    println("max_LB: ", max_LB)
                end
            end
            # continue   
        else
            @timeit get_timer("Shared") "SelectVardMaxLBCenterRange" begin
                if method == "BB+Basic"
                    bVarIdx, bVarIdy = branch.SelectVardMaxLBCenterRange(group_centers)
                else
                    if rank == p_root
                        bVarIdx, bVarIdy = branch.SelectVarMaxRange(node)
                    else
                        bVarIdx = nothing
                        bVarIdy = nothing
                    end
                    MPI.Barrier(comm)
                    bVarIdx = MPI.bcast(bVarIdx, p_root, comm)
                    bVarIdy = MPI.bcast(bVarIdy, p_root, comm)
                    MPI.Barrier(comm)
                end 
            end
            if rank == p_root
                println("branching on ", bVarIdx," dimension,    ", bVarIdy, " th cluster")
            end
            @timeit get_timer("Shared") "branch!" begin
                if rank == p_root
                    bValue = (node.upper[bVarIdx,bVarIdy] + node.lower[bVarIdx,bVarIdy])/2;
                    del_num = branch!(X, nodeList, LB_List, bVarIdx, bVarIdy, bValue, node, node_LB, k, check, method); 
                else
                    del_num = nothing
                end
                if world_size > 1
                    MPI.Barrier(comm)
                    del_num = MPI.bcast(del_num, p_root, comm)
                    MPI.Barrier(comm)
                end
            end # end for begin
        end
        del_nums += del_num
        end # end for begin

        if rank == p_root
            println("Free_memory: ", Sys.free_memory()/2^20)
        end
        if Sys.free_memory() / Sys.total_memory() < 0.1
            @timeit get_timer("Shared") "GC" GC.gc()
        end
        if gc_flag && iter % 1000 == 0
            @timeit get_timer("Shared") "GC" GC.gc()
        end
    end # end for while
    
    ###############################
    if rank == p_root
        if nodeList==[]
            if rank == p_root
                println("\n all node solved")
            end
            # save final calcuation information
            push!(calcInfo, [iter, length(nodeList), 0, max_LB, UB, (UB-max_LB)/min(abs(max_LB), abs(UB))])
        else
            max_LB = calcInfo[end][4]
        end
        println("solved nodes:  ",iter)
        println("deleted nodes:  ",del_nums)
        @printf "%-52d  %-10.4f %-10.4f %-7.4f %s \n" iter  max_LB UB (UB-max_LB)/min(abs(max_LB),abs(UB))*100 "%"

        medoids_index = centers2index(X, centers)
        println("index: ", medoids_index)
        println("centers: ", centers)
        println("centers2: ", X[:, medoids_index])

        r_finish_time = time_ns()
        println("time_comsumed in all the nodes (s) at rank $rank: $(round(Int, (r_finish_time-r_start_time) / (10^9)))\n")
        println("max_trial in BB+LD: ", LD_trial)
        println("FBBT - All samples in kernel region: $(kernel_delta[1]), delta sample number: $(kernel_delta[2])")
    else
        calcInfo = nothing
    end
    MPI.Barrier(comm)
    calcInfo = MPI.bcast(calcInfo, p_root, comm)
    MPI.Barrier(comm)

    return centers, UB, calcInfo
end # end for function

function centers2index(X, centers)
    d, k = size(centers)
    d, s = size(X)
    medoids_index = zeros(Int64, k)
    for i in 1:k
        for j in 1:s
            if sum(X[:, j] - centers[:, i]) == 0
                medoids_index[i] = j
                break
            end
        end
    end

    return medoids_index
end

# end of the module
end 

