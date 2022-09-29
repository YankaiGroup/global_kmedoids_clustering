module fbbt

using branch
using Distances
using TimerOutputs: @timeit, get_timer
using MPI
using SparseArrays


export medoids_bound_tighen, assign_check

# function to calcuate the median value of a vector with 3 elements
function med(a::T, b::T, c::T) where {T <: Union{Float64, Matrix{Float64}}}
    return a + b + c - max(a, b, c) - min(a, b, c)
end

# Check assiged samples according to max and min distance
function assign_check(X::Matrix{Float64}, k::Int, lower::T = nothing, upper::T = nothing, old_check::T2 = nothing) where {T<:Union{Nothing, Matrix{Float64}}, T2<:Union{Nothing, SparseVector{Int, Int}}}
    #@timeit get_timer("Shared") "probing and assign checking" begin
    d, n = size(X)
    if lower === nothing
        lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    end
    if old_check === nothing
        old_check = spzeros(Int, n) # each element is the fixed cluster index, if zero, then this element is not determined
    end
    #end
    check = copy(old_check)
    # distance of x_s to its closest mu
    for s in 1:n::Int
        if old_check[s] == 0 # only update the samples that is not determined yet
            # the way median is precalculated is faster 
            #@timeit get_timer("Shared") "median" begin
            mu = med.(lower, repeat(X[:,s], 1, k), upper) # solution for each scenario
            #end
            #@timeit get_timer("Shared") "colwise dist" begin
            dist_all = colwise(SqEuclidean(), X[:,s], mu)
            #end
            #@timeit get_timer("Shared") "partialsort" begin
            idx = partialsortperm(dist_all, 1:2) # exist bugs when k = 1
            #end
            # max dist calcuation
            #for i in 1:k
            # dist1 must smaller than dist2 to otherwise s can not be determined
            if dist_all[idx[1]] + 1e-15 <= dist_all[idx[2]] 
                max_dist = 0
                for j in 1:d::Int
                    dxi = max(upper[j,idx[1]]-X[j,s], X[j,s]-lower[j,idx[1]])
                    max_dist += dxi^2
                end
                if (max_dist + 1e-6) <=  dist_all[idx[2]]
                    check[s] = idx[1]
                end
            end
        end
    end
    return check
end

function getLowerBound_analytic_fbbt(X::Matrix{Float64}, k::Int, lower::T = nothing, upper::T = nothing) where {T <: Union{Nothing, Matrix{Float64}}}
    d, n = size(X)
    if lower === nothing
        lower, upper = opt_functions.init_bound(X, d, k, lower, upper)
    end
    # get the mid value of each mu for each cluster
    # start calculating the lower bound (distance of x_s to its closest mu)
    LB = 0
    for s = 1:n::Int
        mu = med.(lower, repeat(X[:,s], 1, k), upper) # solution for each scenario
        LB += minimum(colwise(SqEuclidean(), X[:,s], mu))
    end

    return LB
end

# 1-fixed LB-; 2-dynamic LB-; analytic; one
function fbbt_analytic_2(X, k, kernel_index, assign_idx, UB, fixed_centers, lower, upper, LB) # using TD as LB and analytic but loose lower and upper bounds
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0
    zero_sample_flag = false
    d, n = size(X)

    new_kernel_index = Vector{Vector{Int}}(undef, k)
    n_lower = zeros(d, k)
    n_upper = zeros(d, k)

    kernel_delta = zeros(2) # [all kernel number, delta number]
    for i = 1:k::Int
        tmp_fixed_centers = copy(fixed_centers)
        if tmp_fixed_centers[i] == 0
            idx = findall(x -> x == i, assign_idx) # index of assigned samples
            if length(idx) > 0 && length(idx) < n # exist assigned sample for kth cluster
                if rank == p_root
                    println(i, "th cluster - idx length: ", length(idx))
                end
                uidx = findall(x -> x != i, assign_idx)

                @timeit get_timer("Shared") "calcuate delta" begin
                # calcuate delta
                if length(uidx) <= length(idx)
                    for j = 1:k::Int
                        if tmp_fixed_centers[j] != 0 # jth kernel is fixed
                            tmp_center_index = findall(x -> x == Int(tmp_fixed_centers[j]), uidx)
                            if tmp_center_index != []
                                tmp_fixed_centers[j] = tmp_center_index[1]
                            else
                                push!(uidx, Int(tmp_fixed_centers[j]))
                                tmp_fixed_centers[j] = length(uidx)
                            end
                        end
                    end
                    @timeit get_timer("Shared") "tmp_LB" tmp_LB = getLowerBound_analytic_fbbt(X[:, uidx], k, lower, upper)
                else
                    for j = 1:k::Int
                        if tmp_fixed_centers[j] != 0 # jth kernel is fixed
                            tmp_center_index = findall(x -> x == Int(tmp_fixed_centers[j]), idx)
                            if tmp_center_index != []
                                tmp_fixed_centers[j] = tmp_center_index[1]
                            else
                                push!(idx, Int(tmp_fixed_centers[j]))
                                tmp_fixed_centers[j] = length(idx)
                            end
                        end
                    end
                    @timeit get_timer("Shared") "tmp_LB_2" tmp_LB_2 = getLowerBound_analytic_fbbt(X[:, idx], k, lower, upper)
                    tmp_LB = LB - tmp_LB_2
                end
                delta = UB - tmp_LB 
            end # for begin

            @timeit get_timer("Shared") "fbbt analytic" begin
                # fbbt analyticly
                for j in 1:d::Int # au^2+bu+c=delta
                    fa = length(idx)
                    fb = -2*sum(X[j, idx])
                    fc = sum(X[j, idx].^2)
                    for l in 1:d::Int
                        if l != j
                            tmp_u = sum(X[l, idx])/fa # u at min value
                            tmp_u = med(tmp_u, lower[l, i], upper[l, i]) # u at min value in the region
                            fc = fc + fa*tmp_u^2-2*(sum(X[l, idx]))*tmp_u + sum(X[l, idx].^2)
                        end
                    end
                    fc = fc - delta
                    tmp_delta = fb^2 - 4*fa*fc
                    # print("fa: $fa, fb: $fb, fc: $fc, tmp_delta: $tmp_delta.  ")
                    if tmp_delta >= 0 # exist roots in real set
                        tmp_lower = (-fb - sqrt(tmp_delta))/(2*fa)
                        tmp_upper = (-fb + sqrt(tmp_delta))/(2*fa)
                    else # no roots in real set so return zero_sample_flag = true
                        zero_sample_flag = true
                        println("In solver, $i th cluster - zero sample")
                        break
                    end
                    n_lower[j, i] = tmp_lower
                    n_upper[j, i] = tmp_upper
                end

                # update lower and upper according to new bounds
                new_tmp_kernel_index = Int[]
                tmp_kernel_index = kernel_index[i]
                if !zero_sample_flag
                    for j = 1:d::Int
                        if n_lower[j, i] > upper[j, i] || n_upper[j, i] < lower[j, i]
                            zero_sample_flag = true
                            println("TD as LB - In update, $i th cluster - zero sample")
                            break
                        end
                        if n_lower[j, i] < lower[j, i]
                            n_lower[j, i] = lower[j, i]
                        end
                        if n_upper[j, i] > upper[j, i]
                            n_upper[j, i] = upper[j, i]
                        end
                    end
                    for j in tmp_kernel_index::Vector{Int} # check single sample in the kernel region
                        if sum(view(X,:,j) .>= view(n_lower,:,i))==d && sum(view(X,:,j) .<= view(n_upper,:,i))==d
                            push!(new_tmp_kernel_index, j)
                        end
                    end
                    if length(new_tmp_kernel_index) == 0
                        zero_sample_flag = true
                        if rank == p_root
                            println("TD as LB - In update, $i th cluster - zero sample")
                        end
                    end
                end
                new_kernel_index[i] = new_tmp_kernel_index

                tmp_kernel_delta = length(tmp_kernel_index) - length(new_tmp_kernel_index)
                kernel_delta[2] += tmp_kernel_delta
                kernel_delta[1] += length(tmp_kernel_index)
                if rank == p_root
                    println("TD as LB - Kernel ", i, " : Before FBBT :", length(tmp_kernel_index), ", after: ", length(new_tmp_kernel_index), ", delta: ", length(tmp_kernel_index) - length(new_tmp_kernel_index))
                end
            end # for begin
            else # no sample is assgined for this kernel
                n_lower[:, i] = lower[:, i] 
                n_upper[:, i] = upper[:, i]
                new_kernel_index[i] = kernel_index[i]
            end
        else # ith center is fixed
            n_lower[:, i] = X[:, Int(fixed_centers[i])]
            n_upper[:, i] = X[:, Int(fixed_centers[i])]
            new_kernel_index[i] = [Int(fixed_centers[i])]
        end
    end

    return new_kernel_index, n_lower, n_upper, kernel_delta, zero_sample_flag
end

function fbbt_tighter_analytic(X, dmat, k, kernel_index, assign_idx, UB, glbLB, lambda, alpha, groups, alpha_index_k, lower, upper) # using LD as LB
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0
    if rank == p_root
        d, ~ = size(X)
    else
        d = nothing
    end
    MPI.Barrier(comm)
    d = MPI.bcast(d, p_root, comm)
    MPI.Barrier(comm)

    n_rank = length(groups)
    zero_sample_flag = false

    new_kernel_index = Vector{Vector{Int}}(undef, k)
    n_lower = zeros(d, k)
    n_upper = zeros(d, k)

    kernel_delta = zeros(2) # [all kernel number, delta number]

    #### update of medoids in the child node ####
    kernel_num = zeros(Int, k)
    for i in 1:k::Int
        kernel_num[i] = length(kernel_index[i])
    end
    old_alpha_index_k = copy(alpha_index_k)
    alpha_index_k = zeros(Int, k)
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
            alpha_index_k[i] = old_alpha_index_k[i]
        end
    end
    if rank == p_root
        println("old_alpha_index_k: ", old_alpha_index_k)
        println("alpha_index_k: ", alpha_index_k)
    end

    #### update of bounds in the child node ####
    for i = 1:k::Int
        idx = findall(x -> x == i, assign_idx)
        if length(idx) > 0 # exist assigned sample for kth cluster
            @timeit get_timer("Shared") "delta" begin
                if rank == p_root
                    println("LD as LB - fbbt_tighter_$i - length(idx): ", length(idx))
                end
                # calcuate delta
                tmp_LB = glbLB
                tmp_LB = tmp_LB - sum(lambda[idx])
                overall_fixed_alpha_sum = 0
                for m in 1:k::Int
                    n_center_rank = nothing
                    n_center = nothing
                    for j in 1:n_rank::Int
                        n_center = findall(x->x==alpha_index_k[m], groups[j])
                        if n_center != []
                            n_center = n_center[1]
                            n_center_rank = j-1
                            break
                        end
                    end

                    if rank == n_center_rank # the process containing alpha_index_k[i]
                        fixed_alpha_sum = 0
                        for l in idx::Vector{Int} # dmat: n * group_num
                            fixed_alpha_sum += min(0, dmat[l, n_center] - lambda[l])
                        end
                    else
                        fixed_alpha_sum = nothing
                    end
                    MPI.Barrier(comm)
                    fixed_alpha_sum = MPI.bcast(fixed_alpha_sum, n_center_rank, comm)
                    MPI.Barrier(comm)
                    overall_fixed_alpha_sum += fixed_alpha_sum
                end
                    
                tmp_LB = tmp_LB - overall_fixed_alpha_sum
                delta = UB - tmp_LB
            end # for begin

            # fbbt
            @timeit get_timer("Shared") "fbbt analytic" begin
                if rank == p_root
                    # fbbt analyticly
                    for j in 1:d::Int # au^2+bu+c=delta
                        fa = length(idx)
                        fb = -2*sum(X[j, idx])
                        fc = sum(X[j, idx].^2)
                        for l in 1:d::Int
                            if l != j
                                tmp_u = sum(X[l, idx])/fa # u at min value
                                tmp_u = med(tmp_u, lower[l, i], upper[l, i]) # u at min value in the region
                                fc = fc + fa*tmp_u^2-2*(sum(X[l, idx]))*tmp_u + sum(X[l, idx].^2)
                            end
                        end
                        fc = fc - delta
                        tmp_delta = fb^2 - 4*fa*fc
                        # print("fa: $fa, fb: $fb, fc: $fc, tmp_delta: $tmp_delta.  ")
                        if tmp_delta >= 0 # exist roots in real set
                            tmp_lower = (-fb - sqrt(tmp_delta))/(2*fa)
                            tmp_upper = (-fb + sqrt(tmp_delta))/(2*fa)
                        else # no roots in real set so return zero_sample_flag = true
                            zero_sample_flag = true
                            if rank == p_root
                                println("LD as LB - In solver, $i th cluster - zero sample")
                            end
                            break
                        end
                        n_lower[j, i] = tmp_lower
                        n_upper[j, i] = tmp_upper
                    end
                    
                    new_tmp_kernel_index = Int[]
                    tmp_kernel_index = kernel_index[i]
                    if !zero_sample_flag
                        for j = 1:d::Int
                            if n_lower[j, i] > upper[j, i] || n_upper[j, i] < lower[j, i]
                                zero_sample_flag = true
                                if rank == p_root
                                    println("LD as LB - In update, $i th cluster - zero sample")
                                end
                                break
                            end
                            if n_lower[j, i] < lower[j, i]
                                n_lower[j, i] = lower[j, i]
                            end
                            if n_upper[j, i] > upper[j, i]
                                n_upper[j, i] = upper[j, i]
                            end
                        end
                        for j in tmp_kernel_index::Vector{Int} # check existence
                            if sum(view(X,:,j) .>= view(n_lower,:,i))==d && sum(view(X,:,j) .<= view(n_upper,:,i))==d
                                push!(new_tmp_kernel_index, j)
                            end
                        end
                        if length(new_tmp_kernel_index) == 0
                            zero_sample_flag = true
                            if rank == p_root
                                println("LD as LB - In update, $i th cluster - zero sample")
                            end
                        end
                    end
                    new_kernel_index[i] = new_tmp_kernel_index

                    tmp_kernel_delta = length(tmp_kernel_index) - length(new_tmp_kernel_index)
                    kernel_delta[2] += tmp_kernel_delta
                    kernel_delta[1] += length(tmp_kernel_index)
                    println("LD as LB - Kernel ", i, " : Before FBBT :", length(tmp_kernel_index), ", after: ", length(new_tmp_kernel_index), ", delta: ", length(tmp_kernel_index) - length(new_tmp_kernel_index))
                end # if rank == p_root
                MPI.Barrier(comm)
                new_kernel_index = MPI.bcast(new_kernel_index, p_root, comm)
                n_lower = MPI.bcast(n_lower, p_root, comm)
                n_upper = MPI.bcast(n_upper, p_root, comm)
                kernel_delta = MPI.bcast(kernel_delta, p_root, comm)
                zero_sample_flag = MPI.bcast(zero_sample_flag, p_root, comm)
                MPI.Barrier(comm)
            end # for begin
        else # not exist assigned sample for kth cluster
            n_lower[:, i] = lower[:, i] 
            n_upper[:, i] = upper[:, i]
            
            new_kernel_index[i] = kernel_index[i]
        end # if length(kernel_index[i]) > 0
    end # for i = 1:k::Int

    return new_kernel_index, n_lower, n_upper, kernel_delta, zero_sample_flag
end

# lower and upper bounds of medoids tightening for LD
function medoids_bound_tighen(X, k, lower, upper, UB, glbLB, fixed_centers, fbbt_type, LB, assign_idx = nothing, lambda = nothing, alpha = nothing, groups = nothing, alpha_index_k = nothing, dmat = nothing, fbbt_method="fbbt", reduce_alpha = false)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0

    @timeit get_timer("Shared") "check 2" begin
    zero_sample_flag = false
    if rank == p_root
        n_lower, n_upper, kernel_index = CheckAllPoints(X, lower, upper)
    else
        n_lower = nothing
        n_upper = nothing
        kernel_index = nothing
    end
    MPI.Barrier(comm)
    n_lower = MPI.bcast(n_lower, p_root, comm)
    n_upper = MPI.bcast(n_upper, p_root, comm)
    kernel_index = MPI.bcast(kernel_index, p_root, comm)
    MPI.Barrier(comm)
    end # for begin

    @timeit get_timer("Shared") "FBBT" begin
        kernel_delta = zeros(2)
        if fbbt_method != "no_fbbt"
            if sum(assign_idx) != 0 # FBBT
                fbbt_type = false
                if fbbt_type # LD as LB
                    if lambda !== nothing # # fbbt in BB+LD
                        if  fbbt_method == "fbbt_analytic"
                            @timeit get_timer("Shared") "Alt LD as LB" kernel_index, n_lower, n_upper, kernel_delta[1:2], zero_sample_flag = fbbt_tighter_analytic(X, dmat, k, kernel_index, assign_idx, UB, glbLB, lambda, alpha, groups, alpha_index_k, lower, upper) 
                        end
                    end
                else # TD as LB
                    if fbbt_method == "fbbt_analytic"
                        if rank == p_root
                            @timeit get_timer("Shared") "Alt TD as LB" kernel_index, n_lower, n_upper, kernel_delta[1:2], zero_sample_flag = fbbt_analytic_2(X, k, kernel_index, assign_idx, UB, fixed_centers, n_lower, n_upper, LB)
                        else
                            kernel_index = nothing
                            n_lower = nothing
                            n_upper = nothing
                            kernel_delta = nothing
                            zero_sample_flag = nothing
                        end
                        MPI.Barrier(comm)
                        kernel_index = MPI.bcast(kernel_index, p_root, comm)
                        n_lower = MPI.bcast(n_lower, p_root, comm)
                        n_upper = MPI.bcast(n_upper, p_root, comm)
                        kernel_delta = MPI.bcast(kernel_delta, p_root, comm)
                        zero_sample_flag = MPI.bcast(zero_sample_flag, p_root, comm)
                        MPI.Barrier(comm)
                    end
                end
            end
        end
    end # for begin

    return kernel_index, n_lower, n_upper, zero_sample_flag, kernel_delta
end

end # for module
