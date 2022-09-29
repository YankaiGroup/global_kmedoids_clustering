module lb_functions

using Random, Distributions
using LinearAlgebra, Statistics
using Distances
using MPI
using TimerOutputs: @timeit, get_timer

using opt_functions, fbbt, branch, Nodes

export center_index_coor, getLowerBound_adptGp_LD, getLowerBound_analytic, getGlobalLowerBound_up, getLowerBound_analytic_LD, init_analytic_LD_lambda_2, UB_improve, analytic_LD_subgrad


tol = 1e-6
mingap = 1e-3

# function to calcuate the median value of a vector with 3 elements
function med(a::T, b::T, c::T) where {T <: Union{Float64, Matrix{Float64}}}
    return a + b + c - max(a, b, c) - min(a, b, c)
end

# function to transfer centers index to coordinates
function center_index_coor(X::Matrix{Float64}, group_centers_index::Union{Vector{Int64}, Matrix{Int64}})
    d, ~ = size(X)
    num_centers = length(size(group_centers_index))
    # println(group_centers_index)
    if num_centers == 1 # one choice of medoids set
        group_centers_index = Int.(group_centers_index)
        group_centers_coor = X[:, group_centers_index]
    else # multi choice of medoids set
        k, n = size(group_centers_index)
        group_centers_coor = zeros(Int, d, k, n)
        for i = 1:n::Int
            for j = 1:k::Int
                group_centers_coor[:, j, i] = X[:, group_centers_index[j, i]]
            end
        end
    end
    return group_centers_coor
end

############## Lower bound calculation with closed-form ##############
function getLowerBound_analytic(X::Matrix{Float64}, k::Int, lower::T = nothing, upper::T = nothing) where {T <: Union{Nothing, Matrix{Float64}}}
    d, n = size(X)
    # println(size(X))
    if lower === nothing
        lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    end
    # get the mid value of each mu for each cluster
    # start calculating the lower bound (distance of x_s to its closest mu)
    LB = 0
    centers_gp = rand(d, k, n)
    for s = 1:n::Int
        # the way median is precalculated is faster 
        mu = med.(lower, repeat(X[:,s], 1, k), upper) # solution for each scenario
        centers_gp[:,:,s] = mu
        LB += minimum(colwise(SqEuclidean(), X[:,s], mu))
    end
    @timeit get_timer("Shared") "trans in CF" begin
        t_ctr = mean(centers_gp, dims=3)[:,:,1] 
        ~, centers_gp_index = checkcentersample(X, t_ctr, k)
    end
    return LB, centers_gp, centers_gp_index
end

############## Lower bound calculation: closed-form LD from Fisher ##############
function init_analytic_LD_lambda(dmat::Matrix{Float64}, group::Vector{Int}) # min of distances except the distance of oneself
    n, ct = size(dmat)
   
    lambda = zeros(ct) .+ Inf
    for j = 1:ct::Int
        for i = 1:n::Int    
            if i != group[j]
                tmp_d = dmat[i, j]
                if tmp_d < lambda[j]
                    lambda[j] = tmp_d
                end
            end
        end
    end
    return lambda
end

function init_analytic_LD_lambda_2(dmat::Matrix{Float64}, group::Vector{Int}, centers_index::Vector{Int}, assign::Vector{Int}) # local optimal lambda init
    k = size(centers_index)
    n, ct = size(dmat)
    lambda = zeros(Float64, ct)

    for j = 1:ct::Int
       lambda[j] = dmat[centers_index[assign[group[j]]], j]
    end

    return lambda
end

function asgn_lbl(assign::Union{BitArray, Vector{Int}})
    if assign isa BitArray
        return - vec(mapslices(sum, assign, dims=2)).+1
    else
        return - vec(assign).+1
    end
end

# Subgradient Method: periodically update alpha with decay_cycle=20 and initial_alpha=2 [Cornuejols et al., 1977]
function update_analytic_LD_Lambda_subgrad(lambda::Vector{Float64}, UB::Float64, LB::Float64, assign::Union{BitArray, Vector{Int}}, trial::Int, step_rate::Float64, decay_cycle::Int = 20)
    # update lambda
    n = size(lambda)[1]
    # subgradient vector
    S = asgn_lbl(assign)
    new_lambda = zeros(Float64, n)
    alpha = 2 / (2^(floor(Int, trial / decay_cycle))) #5)))
    if alpha < 1e-4 || alpha == Inf
        alpha = 1e-4
    end
    # alpha = 1
    if norm(S)^2 != 0
        current_step_rate = alpha * (UB - LB) / (norm(S)^2) # step rate
    else
        current_step_rate = step_rate
    end
    # println(t)
    for i = 1:n::Int
        new_lambda[i] = max(0, lambda[i] + current_step_rate * S[i])
    end

    return S, alpha, new_lambda, current_step_rate
end

# Subgradient Method: update alpha by red, green method [Escudero et al., 2013]
function update_analytic_LD_Lambda_subgrad_2(lambda::Vector{Float64}, UB::Float64, current_LB::Float64, last_LB::Float64, last_S::Vector{Int}, assign::Union{BitArray, Vector{Int}}, step_rate::Float64, alpha::Float64 = 1.0)
    # update lambda
    n = size(lambda)[1]
    # subgradient vector
    S = asgn_lbl(assign)
    if norm(S)^2 != 0.0
        current_step_rate = alpha * (UB - current_LB) / (norm(S)^2) # step rate
    else
        current_step_rate = step_rate
    end
    new_lambda = zeros(Float64, n)
    for i = 1:n::Int
        new_lambda[i] = max(0, lambda[i] + current_step_rate * S[i])
    end

    # update alpha
    hk = S' * last_S
    if current_LB < last_LB # red
        # print("red ")
        alpha = 0.95 * alpha # here if #red = 1, means once we hit a decrease, we reduce the alpha
    else
        if hk >= 0.0 # green
            # print("green ")
            alpha = 1.1 * alpha
        end # hk <0 is yellow and no change on alpha
    end
    if alpha > 2.0
        alpha = 2.0
    end
    if alpha < 1.0e-4
        alpha = 1.0e-4
    end
    # println(alpha)

    return S, alpha, new_lambda, current_step_rate
end

# Outer entrance to subgradient LD
function getLowerBound_analytic_LD_subgradient(X::Matrix{Float64}, dmat::Matrix{Float64}, k::Int, overall_end_time::Int, groups::Vector{Vector{Int}}, counts::Vector{Int}, node_num::Int, node::Union{Nothing, Node} = nothing, UB::Union{Nothing, Float64} = nothing, lambda_updater::String = "periodical", LD_trial::Int = 30, fbbt_method::String = "fbbt")
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0

    if node.lambda !== nothing
        parent_lambda = copy(node.lambda)
    else
        parent_lambda = node.lambda
    end
    if node.best_lambda !== nothing
        parent_best_lambda = copy(node.best_lambda)
    else
        parent_best_lambda = node.best_lambda
    end
    if node.group_centers !== nothing
        parent_centers = copy(node.group_centers) # initial guess of variable centers for each group
    else
        parent_centers = node.group_centers
    end
    if node.best_alpha_index_k !== nothing
        parent_best_alpha_index_k = copy(node.best_alpha_index_k)
    else
        parent_best_alpha_index_k = node.best_alpha_index_k
    end
    fbbt_type = copy(node.fbbt_type)
    best_alpha = node.best_alpha

    glbLB = node.LB
    r_start_time = time_ns()
    @timeit get_timer("Shared") "medoids_bound_tighen" kernel_index, lower, upper, zero_sample_flag, kernel_delta = medoids_bound_tighen(X, k, node.lower, node.upper, UB, glbLB, node.fixed_centers, fbbt_type, node.CF_LB, node.assign_idx, parent_best_lambda, node.best_alpha, groups, parent_best_alpha_index_k, dmat, fbbt_method, false)

    if zero_sample_flag
        if rank == p_root
            println("zero_sample_flag: $zero_sample_flag")
        end
        return UB*1.5, UB, parent_lambda, parent_best_lambda, parent_centers, parent_best_alpha_index_k, fbbt_type, kernel_delta, best_alpha, node.CF_LB # return UB as LB to prevent further branching of this node
    end

    if rank == p_root
    # lower and upper bounds of medoids tightening and fbbt
        no_kernel_flag = false
        for i = 1:k::Int
            if kernel_index[i] == [] # no sample in ith Kernel
                no_kernel_flag = true
            end
        end
        @timeit get_timer("Shared") "init_LB" CF_LB, centers_gp, centers_gp_index = getLowerBound_analytic(X, k, lower, upper)
    else
        CF_LB = nothing
        centers_gp_index = nothing
        no_kernel_flag = false
    end
    MPI.Barrier(comm)
    CF_LB = MPI.bcast(CF_LB, p_root, comm)
    centers_gp_index = MPI.bcast(centers_gp_index, p_root, comm)
    no_kernel_flag =  MPI.bcast(no_kernel_flag, p_root, comm)
    MPI.Barrier(comm)
    
    if no_kernel_flag # no sample in at least one Kernel
        if rank == p_root
            println("no_kernel_flag: $no_kernel_flag")
        end
        return UB*1.5, UB, parent_lambda, parent_best_lambda, parent_centers, parent_best_alpha_index_k, fbbt_type, kernel_delta, best_alpha, CF_LB # return UB as LB to prevent further branching of this node
    end

    if CF_LB < UB
        @timeit get_timer("Shared") "all_subgrad" LD_LB, LD_UB, lambda, best_lambda, medoids_index, best_alpha_index_k, best_alpha = analytic_LD_subgrad(X, k, overall_end_time, kernel_index, groups, counts, node_num, UB, parent_lambda, parent_best_lambda, node.best_alpha, parent_best_alpha_index_k, glbLB, parent_centers, dmat, lambda_updater, LD_trial)
    else
        if rank == p_root
            println("init_LB from CF > UB - init_LB:", CF_LB)
        end
        return UB*1.5, UB, parent_lambda, parent_best_lambda, parent_centers, parent_best_alpha_index_k, fbbt_type, kernel_delta, best_alpha, CF_LB # return UB as LB to prevent further branching of this node
    end
    r_finish_time = time_ns()

    # maximum(TD, LD)
    if rank == p_root
        group_centers_coor = center_index_coor(X, centers_gp_index)
        CF_UB, ~ = obj_assign(group_centers_coor, X) # opt_functions
    else
        CF_UB = nothing
    end
    MPI.Barrier(comm)
    CF_UB = MPI.bcast(CF_UB, p_root, comm)
    MPI.Barrier(comm)

    # select best UB
    UB, tmp_index = findmin([UB CF_UB LD_UB])
    if tmp_index[2] == 1 # results from parent
        medoids_index = parent_centers 
    elseif tmp_index[2] == 2 # results from CF
        medoids_index = centers_gp_index 
    end

    # select best LB
    LB, tmp_index = findmax([glbLB CF_LB LD_LB])
    if tmp_index[2] == 2 # results from CF
        fbbt_type = false 
    elseif tmp_index[2] == 3 # results from LD
        fbbt_type = true 
    end
    if LB != LD_LB
        lambda = parent_lambda
    end

    gap = (UB - LB) / (LB) * 100
    if rank == p_root
        println("BB+LD result - LB: ", LB, "\t UB: ", UB, "\t gap:", gap, "%")
        println("time_comsumed in one BB+LD (s):", round((r_finish_time - r_start_time) / (10.0^9), digits = 5))
    end

    return LB, UB, lambda, best_lambda, medoids_index, best_alpha_index_k, fbbt_type, kernel_delta, best_alpha, CF_LB
end

# calcuate contribution alpha in subgrad
function alpha_calc_old(group::Vector{Int}, dmat::Matrix{Float64}, lambda::Vector{Float64}, union_kernel_region::BitVector)
    n, ~ = size(dmat)
    group_num = size(group)[1]
    alpha = zeros(group_num)
    for i = 1:n::Int # sample circle # n*n
        if lambda[i] > 0
            for j = 1:group_num::Int # kernel circle
                if union_kernel_region[group[j]] == true
                    alpha[j] += min(0, dmat[i, j] - lambda[i])
                end
            end
        end
    end
    
    return alpha
end

# calcuate contribution alpha in subgrad
function alpha_calc(group::Vector{Int}, dmat::Matrix{Float64}, lambda::Vector{Float64}, union_kernel_region::BitVector)
    n, ~ = size(dmat)
    group_num = size(group)[1]
    alpha = zeros(Float64, group_num)
    @timeit get_timer("Shared") "zero dict" begin
        for j = 1:group_num::Int # sample circle # n*n
            if union_kernel_region[group[j]] == true
                for i = 1:n::Int # kernel circle
                    alpha[j] += min(0.0, dmat[i, j] - lambda[i])
                end
            end
        end
    end # for begin
    return alpha
end

# calculate UB with given minimum alpha indices
function UB_calc(dmat::Matrix{Float64}, alpha_index_k::Vector{Int})
    n = size(dmat)[2] # n*len(gp_k)
    k = length(alpha_index_k)
    current_UB = 0.0 # optimal value of the original problem
    assign = zeros(Int, n)
    for i in 1:n::Int
        tmp_min_d = Inf
        for j = 1:k::Int
            tmp_d = dmat[alpha_index_k[j], i]
            if tmp_d < tmp_min_d
                tmp_min_d = tmp_d
                assign[i] = j
            end
        end
        current_UB += tmp_min_d
    end
    return current_UB, assign
end

# improve the medoids by assignments in K-means like
function UB_improve(X::Matrix{Float64}, assign::Vector{Int}, k::Int, current_medoids_index::Vector{Int}) 
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0

    d, n = size(X)

    medoids = zeros(Float64, d, k)
    for i in unique(assign) # in case of empty clusters
        medoids[:, i] = mean(X[:, findall(x->x==i, assign)], dims = 2)
    end

    new_medoids_index = copy(current_medoids_index)
    ~, medoids_index = checkcentersample(X, medoids, k)
    for i in unique(assign) # in case of empty clusters
        new_medoids_index[i] = medoids_index[i]
    end

    return new_medoids_index
end

# calculate assignment according to the given medoids and lambda, used for subgradient
function assign_calc(dmat::Matrix{Float64}, alpha_index_k::Vector{Int}, lambda::Vector{Float64})
    n = size(dmat)[2]
    k = length(alpha_index_k)
    assign = zeros(Int, n)
    for i in 1:n::Int # groups[g]
        for j = 1:k::Int
            if dmat[alpha_index_k[j],i] <= lambda[i]
                assign[i] += 1
            end
        end
    end
    return assign
end

# inner subgradient for LD
function analytic_LD_subgrad(X::Matrix{Float64}, k::Int, overall_end_time::Int, kernel_index::Vector{Vector{Int}}, groups::Vector{Vector{Int}}, counts::Vector{Int}, node_num::Int, UB::Union{Nothing, Float64} = nothing, lambda::Union{Nothing, Vector{Float64}} = nothing, best_lambda::Union{Nothing, Vector{Float64}} = nothing, 
best_alpha::Union{Nothing, Vector{Float64}} = nothing,best_alpha_index_k::Union{Nothing, Vector{Int}} = nothing, LB::Union{Nothing, Float64} = nothing, medoids_index = nothing, dmat::Matrix{Float64} = nothing, lambda_updater::String = "periodical", maxtrial_no_improve::Int = 3, maxtrial::Int = 10)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0

    # initialize
    trial = 1
    trial_no_improve = 0

    if rank == p_root
        ~, n = size(X)
    else
        n = nothing
    end
    MPI.Barrier(comm)
    n = MPI.bcast(n, p_root, comm)
    MPI.Barrier(comm)

    if medoids_index === nothing
        medoids_index = zeros(Int, k)
    end

    if best_lambda !== nothing
        best_lambda = copy(best_lambda)
    else
        best_lambda = best_lambda
    end

    if best_alpha !== nothing
        best_alpha = copy(best_alpha)
    else
        best_alpha = best_alpha
    end

    if best_alpha_index_k !== nothing
        best_alpha_index_k = copy(best_alpha_index_k)
    else
        best_alpha_index_k = best_alpha_index_k
    end

    intial_node = false
    if lambda === nothing
        if rank == p_root
            println("init_lambda")
        end
        lambda_lcl = init_analytic_LD_lambda(dmat, groups[rank+1]) # (counts[rank+1], 1)
        lambda = MPI.Allgatherv!(lambda_lcl, VBuffer(similar(lambda_lcl, sum(counts)), counts), comm)
        intial_node = true
    end
    MPI.Barrier(comm)

    if LB === nothing
        LB = -Inf
        last_LB = -Inf
    else
        last_LB = LB
    end
    if UB === nothing
        UB = +Inf
    end
    last_S = ones(Int, n)

    pi = 1.0
    control_gap = 1.0
    step_rate = 1.0

    if rank == p_root
        # feasible medoids region
        union_kernel_region = falses(n)
        union_kernel_index = kernel_index[1]
        kernel_num = zeros(Int, k)
        for i in 1:k::Int
            union_kernel_index = union(union_kernel_index, kernel_index[i])
            kernel_num[i] = length(kernel_index[i])
        end
        union_kernel_region[union_kernel_index] .= true
        println("Kernel point number: ", kernel_num)
    else
        union_kernel_region = nothing
    end
    union_kernel_region = MPI.bcast(union_kernel_region, p_root, comm)
    MPI.Barrier(comm)

    break_from_root = false
    # calculate the lower bound with closed-form LD from Fisher
    while trial <= maxtrial && time_ns() <= overall_end_time && trial_no_improve < maxtrial_no_improve
        # determine the feasible LD solution under current lambda
        ############### below we need MPI ################
        @timeit get_timer("Shared") "alphas in subgrad" begin
        alpha_gp = alpha_calc(groups[rank+1], dmat, lambda, union_kernel_region)

        if rank == p_root
            alpha = MPI.Gatherv!(alpha_gp, VBuffer(similar(alpha_gp, sum(counts)), counts), p_root, comm)
        else
            MPI.Gatherv!(alpha_gp, nothing, p_root, comm) 
        end
        MPI.Barrier(comm)
        end # for begin
        ############### above we need MPI ################
        alpha_index_k = zeros(Int, k)
        current_LB = nothing
        if rank == p_root
            # determine the smallest alpha in each kernel region
            alpha_k = zeros(k)  # number of k smallest contribution in ith kernel_region
             # dataset index of kernel
            for i = 1:k::Int
                kernel_num_i = kernel_num[i] 
                if kernel_num_i > k
                    kernel_num_i = k
                end
                alpha_k = partialsortperm(alpha[kernel_index[i]], 1:kernel_num_i) # number of k mallest contribution in ith kernel_region
                tmp_index_k = kernel_index[i][alpha_k] # dataset index of k mallest contribution in ith kernel_region

                for j = 1:kernel_num_i::Int
                    if findall(x -> x == tmp_index_k[j], alpha_index_k) == [] # find the smallest kernel that not selected
                        alpha_index_k[i] = tmp_index_k[j]
                        break
                    end
                end
                if alpha_index_k[i] == 0 # no medoid selected in ith kernel region ??? 
                    #println("no medoid selected in", i, "th kernel region")
                    LB = UB*1.5
                    return LB, UB, lambda, best_lambda, medoids_index, best_alpha_index_k, best_alpha
                end
            end
            current_LB = sum(alpha[alpha_index_k]) + sum(lambda) # optimal value of LD

            # determine the cost of the feasible solution and check the improvments
            current_medoids_index = alpha_index_k
            current_medoids = X[:, alpha_index_k]

            # improve the UB by assignments and kmeans-like centers
            @timeit get_timer("Shared") "UB_improve" begin
                current_UB, current_assign = obj_assign(current_medoids, X)
                if node_num < 10
                    kmeans_count_max = 5
                else
                    kmeans_count_max = 1
                end
                current_UB_org = current_UB
                kmeans_count = 1
                continue_no_improve = 0
                while kmeans_count <= kmeans_count_max && continue_no_improve < 3 
                    new_alpha_index_k = UB_improve(X, current_assign, k, current_medoids_index)
                    new_current_UB, new_assign = obj_assign(X[:, new_alpha_index_k], X)
                    if new_current_UB < UB
                        current_UB = new_current_UB
                        current_medoids_index = new_alpha_index_k
                        current_medoids = X[:, new_alpha_index_k]
                        continue_no_improve = 0
                    else
                        continue_no_improve += 1
                    end
                    current_assign = new_assign
                    kmeans_count += 1
                end
                if current_UB < current_UB_org
                    println(kmeans_count, " - before UB improvement: ", current_UB_org, ", after UB improvement: ", current_UB, ", gloabl UB: ", UB)
                end
            end # for begin


            # update of UB and LB under current lambda
            if current_UB < UB
                UB = current_UB
                medoids = current_medoids
                medoids_index = current_medoids_index
            end

            if LB < current_LB
                tmp_min_gap = 1e-6
                if (current_LB - LB) >= max(tmp_min_gap, tmp_min_gap*abs(UB), 0.1*(abs(UB)-LB))
                    trial_no_improve = 0
                end
                best_alpha_index_k = copy(alpha_index_k)
                best_lambda = copy(lambda)
                best_alpha = copy(alpha)
                LB = current_LB
            end
        
            # terminal conditions
            trial += 1
            trial_no_improve += 1
            control_gap = (UB - LB) / min(abs(LB), abs(UB))
            if intial_node
                if control_gap < 1e-3
                    println("On ", trial, "th trial: control_gap < 1e-3:   LB:", LB, ",  UB:", UB, ",  control_gap(%):", control_gap*100)
                    break_from_root = true
                end
                intial_node = false
            else
                if LB > UB + 1e-6
                    println("On ", trial, "th trial: LB > UB + 1e-6:   LB:", LB, ",  UB:", UB, ",  control_gap(%):", control_gap*100)
                    break_from_root = true
                end
            end
            if trial_no_improve >= maxtrial_no_improve
                println("On ", trial, "th trial: trial_no_improve > ", maxtrial_no_improve, ":   LB:", LB, ",  UB:", UB, ",  control_gap(%):", control_gap*100)
                break_from_root = true
            end

            if trial > maxtrial
                    println("On ", trial, "th trial: trial > ", maxtrial, ": LB:", LB, ",  UB:", UB, ",  control_gap(%):", control_gap*100)
                break_from_root = true
            end

            if time_ns() >= overall_end_time
                    println("On ", trial, "th trial: overall time out:   LB:", LB, ",  UB:", UB, ",  control_gap(%):", control_gap*100)
                break_from_root = true
            end
        end # if rank==p_root
        MPI.Barrier(comm)
        alpha_index_k = MPI.bcast(alpha_index_k, p_root, comm)
        current_LB = MPI.bcast(current_LB, p_root, comm)
        if rank != p_root
            trial += 1
            trial_no_improve += 1
        end
        trial_no_improve = MPI.bcast(trial_no_improve, p_root, comm)
        break_from_root = MPI.bcast(break_from_root, p_root, comm)
        MPI.Barrier(comm)
        
        # update assignments under current lambda 
        assign_lcl = assign_calc(dmat, alpha_index_k, lambda[groups[rank+1]]) # 
        assign = MPI.Allgatherv!(assign_lcl, VBuffer(similar(assign_lcl, sum(counts)), counts), comm)
        MPI.Barrier(comm)
        # update lambda
        if lambda_updater == "periodical" # original updater
            S, pi, lambda, step_rate = update_analytic_LD_Lambda_subgrad(lambda, UB, current_LB, assign, trial, step_rate, 20) # subgrad
        elseif lambda_updater == "green" # green and red updater
            @timeit get_timer("Shared") "green updater" S, pi, lambda, step_rate = update_analytic_LD_Lambda_subgrad_2(lambda, UB, current_LB, last_LB, last_S, assign, step_rate, pi)
        end
        # update lambda params
        last_S = S
        last_LB = current_LB

        if break_from_root
            LB = MPI.bcast(LB, p_root, comm)
            UB = MPI.bcast(UB, p_root, comm)
            best_alpha_index_k = MPI.bcast(best_alpha_index_k, p_root, comm)
            best_lambda = MPI.bcast(best_lambda, p_root, comm)
            medoids_index = MPI.bcast(medoids_index, p_root, comm)
            MPI.Barrier(comm)
            return LB, UB, lambda, best_lambda, medoids_index, best_alpha_index_k, best_alpha
        end
    end # while
end # function

# end of the module
end

