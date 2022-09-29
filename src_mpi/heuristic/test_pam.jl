# PAM from https://github.com/mthelm85/PAM.jl written by Matt Helm <mthelm85@gmail.com> and other contributors

using Distances
using RDatasets, DataFrames, CSV
using Random, Distributions
#using Plots#, StatsPlots
using MLDataUtils, Clustering
using JLD, Statistics

using TimerOutputs: @timeit, get_timer

# arg1=: number of clusters to be solved
# arg2=: dataset name

# load functions for branch&bound and data preprocess from self-created module
if !("src_mpi/" in LOAD_PATH)
    push!(LOAD_PATH, "src_mpi/")
end

using data_process

############# auxilary functions #############
struct PAMResult
    medoids::Vector{Int}       # cluster medoids (d x k)
    assignments::Vector{Int}        # assignments (n)
    costs::Vector{Float64}          # cost of the assignments (n)
    totalcost::Float64              # total cost (i.e. objective)
end

function build_phase(D, k)
    N = size(D,1)
    # println(N)
    total_dists = [sum(D[:,j]) for j in 1:N]

    # initialize medoids with index of object with shortest distance to all others
    medoids = Int[findmin(total_dists)[2]]
    for j in 1:k-1 
        TD = Vector{Float64}(undef,N)
        for a in 1:N
            td = 0.0
            for i in 1:N
                td += reduce(min, (D[i,m] for m in medoids), init=D[a,i])
            end
            TD[a] = td
        end
        push!(medoids, findmin(TD)[2])
    end

    return medoids
end

function swap_phase(D, k, M)
    n = size(D, 1)
    M = copy(M)
    Mⱼ = similar(M)
    costs = Vector{Float64}(undef, n)
    # Perform clustering
    assignments = Int[findmin(view(D, i,M))[2] for i in axes(D,1)]
    cumulative = similar(assignments)
    while true
        # Find minimum sum for each cluster (i.e. find the best medoid)
        for i in 1:k
            cluster = assignments .== i
            cumsum!(cumulative, cluster)
            D_slice = view(D, cluster, cluster)
            distances = sum(@view D_slice[:,i] for i in 1:last(cumulative))
            smallest_distance_idx = findmin(distances)[2]
            Mⱼ[i] = findfirst(==(smallest_distance_idx), cumulative)::Int
        end 
        if sort(M) == sort(Mⱼ)
            for i in 1:n
                costs[i] = D[i, M[assignments[i]]]
            end
            totalcost = sum(costs)
            return PAMResult(M, assignments, costs, totalcost)
        else
            M,Mⱼ = Mⱼ,M
        end
    end
end

function pam(D, k)
    M = build_phase(D, k)
    return swap_phase(D, k, M)
end

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

#############################################################
################# Main Process Program Body #################
#############################################################
const to = get_timer("Shared")

# real world dataset testing
@timeit to "load data" begin
    dataname = ARGS[2]
    if dataname == "iris"
        X = data_preprocess("iris") # read iris data from datasets package
    else
        if Sys.iswindows()
            X = data_preprocess(dataname, nothing, joinpath(@__DIR__, "..\\..\\data\\"), "NA") # read data in Windows
        else
            X = data_preprocess(dataname, nothing, joinpath(@__DIR__, "../../data/"), "NA") # read data in Mac
        end
    end


    k = parse(Int, ARGS[1]) #length(unique(label))
    Random.seed!(123)
    println("data size: ", size(X))
    println("data type: ", typeof(X))
end # end for begin

@timeit to "pam" begin
    dmat = pairwise(SqEuclidean(), X, dims=2)
    pam_result = pam(dmat, k)    
    # medoids_index = centers2index(X, pam_result.medoids)
    println("PAM - medoids_index: ", pam_result.medoids)
    println("PAM - medoids: ", X[:, pam_result.medoids])
    println("PAM - UB: ", pam_result.totalcost)
end # end for begin

show(to)
print("\n")

# ###### PAM with dynamic distance calcuation for large datasets exceeding ram limits ######
# function build_phase_dynamic(data, k)
#     N = size(data,2)
#     # println(N)
#     total_dists = [sum(pairwise(SqEuclidean(), data[:, j]', data', dims=1)) for j in 1:N] # D[:,j]

#     # initialize medoids with index of object with shortest distance to all others
#     medoids = Int[findmin(total_dists)[2]]
#     for j in 1:k-1 
#         TD = Vector{Float64}(undef,N)
#         for a in 1:N
#             td = 0.0
#             for i in 1:N
#                 td += reduce(min, (sqeuclidean(data[:, i], data[:, m]) for m in medoids), init=sqeuclidean(data[:, a], data[:,i]))
#             end
#             TD[a] = td
#         end
#         push!(medoids, findmin(TD)[2])
#     end

#     return medoids
# end

# function swap_phase_dynamic(data, k, M)
#     n = size(data, 2)
#     M = copy(M)
#     Mⱼ = similar(M)
#     costs = Vector{Float64}(undef, n)
#     # Perform clustering
#     assignments = zeros(Int64, n)
#     for i in 1:n
#         assignments[i] = findmin(pairwise(SqEuclidean(), data[:, i]', data[:, M]', dims = 1))[2][2]
#     end
#     cumulative = similar(assignments)
#     trial = 0
#     while true
#         print(trial, ", ")
#         trial += 1
#         # Find minimum sum for each cluster (i.e. find the best medoid)
#         for i in 1:k
#             cluster = assignments .== i
#             cluster_index = Array(1:n)[cluster]
#             cumsum!(cumulative, cluster)
#             distances = [sum(pairwise(SqEuclidean(), data[:, cluster]', data[:, cluster_index[l]]', dims=1) for l in 1:last(cumulative))]
#             smallest_distance_idx = findmin(distances[1])[2][1]
#             Mⱼ[i] = findfirst(==(smallest_distance_idx), cumulative)::Int
#             # println(distances)
#             # println(smallest_distance_idx)
#             # println(Mⱼ[i])
#         end 
#         if sort(M) == sort(Mⱼ)
#             for i in 1:n
#                 costs[i] = sqeuclidean(data[:, i], data[:, M[assignments[i]]])
#             end
#             totalcost = sum(costs)
#             return PAMResult(M, assignments, costs, totalcost)
#         else
#             M,Mⱼ = Mⱼ,M
#         end
#     end
# end

# function pam_dynamic(data, k)
#     M = build_phase_dynamic(data, k)
#     return swap_phase_dynamic(data, k, M)
# end
