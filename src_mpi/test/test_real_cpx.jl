using RDatasets, DataFrames, CSV
using Random, Distributions, Distances
#using Plots#, StatsPlots
using MLDataUtils, Clustering
using JLD

using TimerOutputs: @timeit, get_timer

# arg1=: number of clusters to be solved
# arg2=: dataset name






# load functions for branch&bound and data preprocess from self-created module
if !("src_mpi/" in LOAD_PATH)
    push!(LOAD_PATH, "src_mpi/")
end

using data_process, bb_functions, opt_functions


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
#println("data size: ", size(data))
# println("data type: ", typeof(data))

k = parse(Int, ARGS[1]) #length(unique(label))
Random.seed!(123)
end # end for begin

@timeit to "CPX" begin
    t_g = @elapsed centers_g, objv_g, iter_g, gap_g = global_OPT_base_dynamic(X, k)
    println("$dataname:\t", round(objv_g, digits=2), "\t", round(t_g, digits=2), "\t", 
        round(gap_g, digits=4), "%\t", iter_g)
end # end for begin

if rank == 0
    show(to)
    print("\n")
end