using RDatasets, DataFrames, CSV
using Random, Distributions, Distances
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
function init_population_kmeans(data, k)
    seeds = [111, 222, 333, 444, 555, 666, 777, 888, 999, 123, 234, 345, 456, 567, 678, 789, 890, 134, 256, 378, 910, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 1345, 2456, 3567, 4678, 5789, 6790, 7891, 8901, 1346, 2457, 3568, 4679, 5790, 6791, 7892, 8902, 1347, 2458, 3569, 4680, 5791, 6792, 7893, 8903, 1348, 2459, 3570, 4681, 5792, 6793, 7894, 8904, 1349, 2460, 3571, 4682, 5793, 6794, 7895, 8905, 1350, 2461, 3572, 4683, 5794, 6795, 7896, 8906, 1351, 2462, 3573, 4684, 5795, 6796, 7897, 8907, 1352, 2463, 3574, 4685, 5796, 6797, 7898]
    d, n = size(data)
    iters = 20
    kmeans_UBs = zeros(iters)
    kmeans_centers_coors = zeros(iters, d, k)
    kmed_UBs = zeros(iters)
    kmed_centers_coors = zeros(iters, d, k)
    kmed_centers_indexs = zeros(Int, iters, k)
    for i in 1:iters
        Random.seed!(seeds[i])
        # kmeans
        clst_rlt = kmeans(data, k, init=:kmpp)
        kmeans_centers_coors[i, :, :] = clst_rlt.centers
        kmeans_UBs[i] = clst_rlt.totalcost
        # kmedoids
        node_centers, group_centers_index = checkcentersample(data, kmeans_centers_coors[i, :, :], k)
        node_UB, ~ = obj_assign(node_centers, data)
        group_centers_index = sort(group_centers_index)
        kmed_centers_coors[i, :, :] = node_centers
        kmed_UBs[i] = node_UB
        kmed_centers_indexs[i, :] = group_centers_index
    end

    return kmeans_UBs, kmeans_centers_coors, kmed_UBs, kmed_centers_coors, kmed_centers_indexs
end

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

@timeit to "kmeans" begin
    kmeans_UBs, kmeans_centers_coors, kmed_UBs, kmed_centers_coors, kmed_centers_indexs = init_population_kmeans(X, k)    


    println("kmeans++ - best coor: ", kmeans_centers_coors[argmax(kmeans_UBs), :, :])
    println("kmedoids - best coor: ", kmed_centers_coors[argmax(kmed_UBs), :, :], ", best_index: ", kmed_centers_indexs[argmax(kmed_UBs), :])
    println("#######")
    println("kmeans++ - max: ", maximum(kmeans_UBs), ", min: ", minimum(kmeans_UBs), ", avg: ", mean(kmeans_UBs))
    println("kmedoids - max: ", maximum(kmed_UBs), ", min: ", minimum(kmed_UBs), ", avg: ", mean(kmed_UBs))

end # end for begin

show(to)
print("\n")