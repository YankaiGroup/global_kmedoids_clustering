# This is the file for precompiling self-created packages in niagara system

if !("src_mpi/" in LOAD_PATH)
    push!(LOAD_PATH, "src_mpi/")
end

# using Nodes, data_process, grb_env, probing, branch, obbt
using bb_functions, lb_functions, ub_functions, opt_functions, data_process, Nodes, branch