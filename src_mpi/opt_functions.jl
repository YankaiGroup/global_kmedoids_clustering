module opt_functions

using Clustering
using Printf
using Random
using Distances
using Metaheuristics
using SparseArrays
using JuMP
using CPLEX
using MathOptInterface
const MOI = MathOptInterface
using TimerOutputs: @timeit, get_timer

export obj_assign, init_ub_heuristic, checkcentersample, global_OPT_base_dynamic

time_lapse = 60*60*4 # 60*60*4h

############# auxilary functions #############
function checkcentersample(X::Matrix{Float64}, t_ctr::Matrix{Float64}, k::Int)
    dmat = pairwise(SqEuclidean(), X, t_ctr, dims = 2) # n * k
    ~, tmp_ctr = findmin(dmat, dims = 1) # find the closest point to kernel
    c_ctr = zeros(Int, k)
    for i in 1:k::Int
	    c_ctr[i] = Int(tmp_ctr[i][1])
    end
    t_ctr = X[:, c_ctr]
    return t_ctr, c_ctr # coordinate, index
end

function obj_assign(centers::Matrix{Float64}, X::Matrix{Float64}, dmat::Union{Nothing, Matrix{Float64}}=nothing)
    d, n = size(X)   	 
    k = size(centers, 2)	 

    if dmat === nothing
        dmat = pairwise(SqEuclidean(), centers, X, dims = 2) # k*n
    end

    costs, a = findmin(dmat, dims = 1) # find the closest cluster that point j belongs to
    assign = [a[j][1] for j in 1:n]
    return sum(costs), assign # sum costs is the total sse, assign is the current clustering assignment
end

# update or get the center lower and update bounds
function init_bound(X::Matrix{Float64}, d::Int, k::Int, lower::T=nothing, upper::T=nothing) where {T<:Union{Nothing, Matrix{Float64}}}
    lower_data = Vector{Float64}(undef, d)
    upper_data = Vector{Float64}(undef, d)
    for i = 1:d::Int # get the feasible region of center
        lower_data[i] = minimum(X[i,:]) # i is the row and is the dimension 
        upper_data[i] = maximum(X[i,:])
    end
    lower_data = repeat(lower_data, 1, k) # first arg repeat on row, second repeat on col
    upper_data = repeat(upper_data, 1, k)

    if lower === nothing
        lower = lower_data
        upper = upper_data
    else
        lower = min.(upper.-1e-4, max.(lower, lower_data))
        upper = max.(lower.+1e-4, min.(upper, upper_data))
    end
    return lower, upper
end

# initilize the population by kmeans
function init_population_kmeans(data, k)
    seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999, 123, 234, 345, 456, 567, 678, 789, 890, 134, 256, 378, 910]
    best_UB = Inf
    best_index = zeros(Int, k)
    indices = Array{Vector{Int}}(undef, 20)
    d, n = size(data)
    for i in 1:20
        Random.seed!(seeds[i])
        clst_rlt = kmeans(data, k)
        km_centers = clst_rlt.centers
        node_centers, group_centers_index = checkcentersample(data, km_centers, k)
        node_UB, ~ = obj_assign(node_centers, data)
        println("node_UB: ", node_UB)
        group_centers_index = sort(group_centers_index)
        if node_UB < best_UB
            best_UB = node_UB
            best_index = group_centers_index
        end
        indices[i] = group_centers_index
    end
    indices = unique(indices)

    println("Kmeans - best_UB: ", best_UB, ", best_index: ", best_index, ", best_center: ", data[:, best_index])

    return best_UB, best_index, indices
end

# heuristic search
function init_ub_heuristic(data, k)
    d, n = size(data)

    # cost function for heuristic
    function f(X)
        # X[1:k] is the cluster center
        dmat = pairwise(SqEuclidean(), data[:, floor.(Int, X)], data, dims = 2) # k*n
        costs, ~ = findmin(dmat, dims = 1)
        fx = sum(costs)
        # inequality constraints LHS <= 0
        gx = Array{Float64}([])
        # zmat[i, j] <= ymat[i]
        for i in 1:k-1
                push!(gx, X[i] + 1 - X[i+1])
        end
    
        # equality constraints LHS == 0
        hx = Array{Float64}([])
    
        return fx, gx, hx
    end

    # initilize the population by kmeans
    @timeit get_timer("Shared") "init kmeans" best_UB, best_index, indices = init_population_kmeans(data, k) # k*n_trial
    bounds = [Array(1:k) n*ones(Int, k)-(k.-Array(1:k))]'
    num_populations = floor(Int, n/10);
    options = Options(iterations=5000)
    if num_populations < 10
        num_populations = 10
    elseif num_populations > 500
        num_populations = 500
        options = Options(iterations=1000, time_limit=120.0)
    end
    X = [bounds[1,:] + i.* ((bounds[2,:] -  bounds[1,:]) / num_populations) for i in 1:num_populations]
    push!(X, bounds[1,:])
    for i in indices
        push!(X, i)
    end
    population = [Metaheuristics.create_child(x, f(x)) for x in X ]
    prev_status = State(Metaheuristics.get_best(population), population)
    

    # heuristic search
    algorithm = ECA(N = length(population), options = options)
    println(algorithm)
    algorithm.status = prev_status;
    result = optimize(f, bounds, algorithm)
    println(result)
    fx = minimum(result)
    x = minimizer(result)
    println("ECA - UB: ", fx, ", index: ", floor.(Int, x))

    return fx, floor.(Int, x)
end

############# global optimization solvers #############
# pure cplex solvers -- original k-medoids problem
function global_OPT_base_dynamic(X, k, lower=nothing, upper=nothing, mute=false)
    @timeit get_timer("Shared") "OPT Part 1" begin
    d, n = size(X)

    @timeit get_timer("Shared") "init_ub_heuristic" begin
        node_UB, group_centers_index = init_ub_heuristic(X, k)
        node_centers = X[:, group_centers_index]
        node_UB, node_assign = obj_assign(node_centers, X)
    end # for begin
    m = Model(CPLEX.Optimizer);
    if mute
        set_optimizer_attribute(m, "CPXPARAM_ScreenOutput", 0)
    end
    # set_optimizer_attribute(m, "CPXPARAM_Threads",1)
    set_optimizer_attribute(m, "CPXPARAM_TimeLimit", 4*60*60) 
    
    ## memory limit
    set_optimizer_attribute(m, "CPXPARAM_MIP_Limits_TreeMemory", 100*1024)
    set_optimizer_attribute(m, "CPXPARAM_WorkMem", 5*1024) # 100*1024 MB for working memory

    ## solve method
    set_optimizer_attribute(m, "CPXPARAM_LPMethod", 4) # 4 - Barrier
    set_optimizer_attribute(m, "CPXPARAM_QPMethod", 4) # 4 - Barrier
    set_optimizer_attribute(m, "CPXPARAM_MIP_Strategy_SubAlgorithm", 4) # 4 - Barrier

    ## set node zero
    set_optimizer_attribute(m, "CPXPARAM_MIP_Strategy_StartAlgorithm", 4) # 0 - auto; 4 - Barrier
    set_optimizer_attribute(m, "CPXPARAM_MIP_Strategy_VariableSelect", 4) # 4 - Branch based on pseudo reduced costs
    set_optimizer_attribute(m, "CPXPARAM_Preprocessing_Presolve", 0) # 0 - Do not apply presolve
    set_optimizer_attribute(m, "CPXPARAM_MIP_Strategy_HeuristicFreq", -1) # -1 - Do not use heuristic
    set_optimizer_attribute(m, "CPXPARAM_Emphasis_MIP", 1) # 1 - Emphasize feasibility over optimality
    # set_optimizer_attribute(m, "CPXPARAM_Preprocessing_Aggregator", 0) # 0 - Do not use any aggregator


    # Variables
    zmat_init = spzeros(Int, n, n)
    for i in 1:n
        zmat_init[i, group_centers_index[node_assign[i]]] = 1
    end
    ymat_init = spzeros(Int, n)
    for i in 1:k
        ymat_init[group_centers_index[i]] = 1
    end
    @variable(m, zmat[i=1:n, j=1:n], Bin, start=zmat_init[i, j])
    @variable(m, ymat[i=1:n], Bin, start=ymat_init[i])
    
    # Constraints
    @constraint(m, [i in 1:n], sum(zmat[i, j] for j in 1:n) == 1);
    @constraint(m, [i in 1:n, j in 1:n], zmat[i, j] <= ymat[j]);
    @constraint(m, sum(ymat[i] for i in 1:n) == k);
    dmat = pairwise(SqEuclidean(), X) # n*n
    # Objective
    # @objective(m, Min, sum(sum(sum((X[:, i] .- X[:, j]).^2) .* zmat[i, j] for i in 1:n) for j in 1:n))
    @objective(m, Min, sum(dmat .* zmat))
    end
    @timeit get_timer("Shared") "OPT Part 2" optimize!(m);
    # println(value.(ymat))
    @timeit get_timer("Shared") "OPT Part 3" begin
    centers_num = findall(isequal(1), value.(ymat))
    println(centers_num)
    centers = X[:, centers_num]
    objv, ~ = obj_assign(centers, X)
    # node = node_count(m)
    node = MOI.get(backend(m), MOI.NodeCount())
    # node = 0 # false node for local enviroment
    gap = relative_gap(m) # get the relative gap for cplex solver
end
    return centers, objv, node, gap
end

end # for module

