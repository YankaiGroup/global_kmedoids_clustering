using RDatasets, DataFrames, CSV
using Random, Distributions, Distances
#using Plots#, StatsPlots
using MLDataUtils, Clustering
using JLD

using TimerOutputs: @timeit, get_timer

# arg1=: number of clusters to be solved
# arg2=: dataset name
# arg3=: lower bound method: CPLEX, BB+TD, LD, BB+LD
# arg4=: check point existence in branch_bound: true, false
# arg5=: fbbt method test: no_fbbt, fbbt_analytic 
# arg6=: maxtrial_no_improve in BB+LD
# arg7=: probing_flag: true, false
# arg8=: gc_flag: true, false

# load functions for branch&bound and data preprocess from self-created module
if !("src_mpi/" in LOAD_PATH)
    push!(LOAD_PATH, "src_mpi/")
end

using data_process, bb_functions, opt_functions
using MPI


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
end # end for begin

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
if rank == 0
    println("data size: ", size(X))
    println("data type: ", typeof(X))
end

@timeit to "Total BB" begin
if ARGS[3] == "CPLEX"
    t_g = @elapsed centers_g, objv_g, iter_g, gap_g = global_OPT_base_dynamic(X, k)
    println("$dataname:\t", round(objv_g, digits=2), "\t", round(t_g, digits=2), "\t", 
        round(gap_g, digits=4), "%\t", iter_g)
else
    check = parse(Bool, ARGS[4])
    lambda_updater = "green"
    fbbt_method = ARGS[5]
    if ARGS[3] == "BB+LD"
        LD_trial = parse(Int, ARGS[6])
    else
        LD_trial = 10000
    end
    probing_flag = parse(Bool, ARGS[7])
    gc_flag = parse(Bool, ARGS[8])

    ## allocate dmat ##
    d, n = size(X);
    world_size = MPI.Comm_size(comm)
    nworkers = world_size #- 1 # in this implementation, we try main processor also participate in worker processing
    groups, group_num = grouping(n, nworkers)
    counts = [length(groups[i]) for i in 1:nworkers::Int]  # used for gather operation
    # once we get grouping scheme, we can have the assignment of groups on all processors
    # this info is mainly used in lower bound calculation
    assign_size = ones(nworkers)  # each worker should have only one subproblem
    accu_size = pushfirst!(accumulate(+, assign_size), 0) # get accumulate size (index) info for each worker
    @timeit get_timer("Shared") "dmat allocation" begin
    # for each worker, calclulate dmat of each subproblem
    dmat = pairwise(SqEuclidean(), X, view(X,:, groups[rank+1]), dims=2) # dmat is n*len(gp_k)
    end
    if rank != 0
        X = Matrix{Float64}(undef, 0, 0)# Matrix{Float64}([0 0; 0 0])
    end
    if rank == 0
        println("Free_memory: ", Sys.free_memory()/2^20)
    end
    GC.gc()
    MPI.Barrier(comm)
    if rank == 0
        println("Free_memory: ", Sys.free_memory()/2^20)
    end
    ## end of allocate dmat ##

    t_adp_LD = @elapsed centers_adp_LD, objv_adp_LD, calcInfo_adp_LD = bb_functions.branch_bound(X, dmat, groups, k, ARGS[3], "fixed", "CPLEX", check, lambda_updater, fbbt_method, LD_trial, probing_flag, gc_flag) #
    if rank == 0
        println("$dataname:\t",  
                round(objv_adp_LD, digits=2), "\t",
                round(t_adp_LD, digits=2), "\t", 
                round(calcInfo_adp_LD[end][end]*100, digits=4), "%\t", 
                calcInfo_adp_LD[end][end] <= 0.001 ? length(calcInfo_adp_LD)-1 : length(calcInfo_adp_LD))
    end
end
end # end for begin

if rank == 0
    show(to)
    print("\n")
end

MPI.Finalize()