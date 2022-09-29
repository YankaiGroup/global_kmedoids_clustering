module probing

using MPI
using fbbt
using TimerOutputs: @timeit, get_timer

export probing_base


function probing_base(X::Matrix{Float64}, k::Int, lower::Matrix{Float64}, upper::Matrix{Float64}, node_LB::Float64, UB::Float64, mingap::Float64, bVarIds::Vector{Int})
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    p_root = 0

    n_steps = 6
    #lower = copy(lower_o)
    #upper = copy(upper_o)	 
    d, n = size(X)
    if rank == p_root
        println("branched on ", bVarIds[1]," dimension,    ", bVarIds[2], " th cluster")
    end

    fathom = false
    if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*min(abs(node_LB), abs(UB))
        if rank == p_root
            println("prob_allrange: analytic LB  ",node_LB, "   >=UB    ", UB)
        end
	    fathom = true 	    
    end
    overall_updated = false
    for trial = 1:2
    	if fathom == true
	        break
    	end
        if trial == 1 || overall_updated == true
            if bVarIds[1] == 0
                dims = 1:d
            else
                dims = bVarIds[1]: bVarIds[1]
            end
            for dim in dims::UnitRange{Int}
                if fathom == true
                    break
                end
                if bVarIds[2] == 0
                    clusters = 1:k
                else
                    clusters = bVarIds[2]: bVarIds[2]
                end
                for clst in clusters::UnitRange{Int}
                    lower_updated = true
                    upper_updated = true
                    step = (upper[dim, clst] - lower[dim, clst])/n_steps
                    for insi = 1:(n_steps-1)::Int
                        if (upper[dim, clst] - lower[dim, clst]) <= (step+1e-6)
                            break
                        end
                        if lower_updated
                            lower_trial = copy(lower)
                            upper_trial	= copy(upper)
                            upper_trial[dim, clst] = lower[dim, clst] + step
                            if dim == 1
                                for j = 1:(k-1)::Int  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                                    if upper_trial[1, k-j] >= upper_trial[1, k-j+1]
                                        upper_trial[1, k-j] = upper_trial[1, k-j+1]
                                    end
                                end
                            end
                            
                            @timeit get_timer("Shared") "TD as LB" LB_trial = fbbt.getLowerBound_analytic_fbbt(X, k,  lower_trial, upper_trial)
                            if (UB-LB_trial)<= mingap || (UB-LB_trial) <= mingap*abs(UB)
                                if rank == p_root
                                    println(trial, "  lower[ ",dim, ",",  clst,"]  from ", lower[dim, clst], "  to ", upper_trial[dim, clst])
                                end
                                lower[dim, clst] = upper_trial[dim, clst] 
                                lower_updated = true
                                overall_updated = true
                                if dim == 1
                                    for j = 2:k::Int  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                                        if lower[1, j] <= lower[1, j-1]
                                            lower[1, j] = lower[1, j-1]
                                        end
                                    end
                                end 
                            else
                                lower_updated = false
                            end
                        end

                        if (upper[dim, clst] - lower[dim, clst]) <=	(step+1e-6)
                            break
                        end
                        if upper_updated
                            lower_trial = copy(lower)
                            upper_trial = copy(upper)
                            lower_trial[dim, clst] = upper[dim, clst] - step
                            if dim == 1
                                for j = 2:k::Int  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                                    if lower_trial[1, j] <= lower_trial[1, j-1]
                                        lower_trial[1, j] = lower_trial[1, j-1]
                                    end
                                end
                            end
                            @timeit get_timer("Shared") "TD as LB" LB_trial = fbbt.getLowerBound_analytic_fbbt(X, k,  lower_trial, upper_trial)

                            if (UB-LB_trial)<= mingap || (UB-LB_trial) <= mingap*abs(UB)
                                if rank == p_root
                                    println(trial, "  upper[ ",dim, ",",  clst,"]  from ", upper[dim, clst], "  to ",	lower_trial[dim, clst])
                                end
                                upper[dim, clst] = lower_trial[dim, clst]
                                upper_updated = true
                                overall_updated = true
                                if dim == 1
                                    for j = 1:(k-1)::Int  # bound tightening avoid symmetric solution, for all feature too strong may eliminate other solution
                                        if upper[1, k-j] >= upper[1, k-j+1]
                                            upper[1, k-j] = upper[1, k-j+1]
                                        end
                                    end
                                end
                            else
                                upper_updated = false
                            end
                        end
                    end

                    if (upper[dim, clst] - lower[dim, clst]) <= (step+1e-6)
                        @timeit get_timer("Shared") "TD as LB" tmp_node_LB = fbbt.getLowerBound_analytic_fbbt(X, k, lower, upper)
                        if tmp_node_LB > node_LB
                            node_LB = tmp_node_LB
                            if (UB-node_LB)<= mingap || (UB-node_LB) <= mingap*abs(UB)
                                if rank == p_root
                                    println("prob_k$clst: analytic LB  ",node_LB, "   >=UB    ", UB)
                                end
                                fathom = true
                                break
                            end
                        end
                    end    
                end
            end
        end
    end
    return lower, upper, node_LB
end


end