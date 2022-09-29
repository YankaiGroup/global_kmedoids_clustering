module data_process

using RDatasets, DataFrames, CSV
using LinearAlgebra

export data_preprocess, sig_gen

# function for data pre-processing, here missingchar will be a single character
function data_preprocess(dataname, datapackage = "datasets", path=nothing, missingchar=nothing, header=false, types=Float64)
    # read data
    if path === nothing # read data from r-package datasets
        data = dataset(datapackage, dataname)
    elseif missingchar === nothing
        println(joinpath(path, dataname))
        data = CSV.read(joinpath(path, dataname), DataFrame, header = header)
    else
        data = CSV.read(joinpath(path, dataname), DataFrame, header = header, types= types, missingstring = missingchar)
        data = dropmissing(data)
    end

    # return data in transpose for optimization process, d * n 
    if dataname == "syn_1E7_2_3"
        return Matrix(Matrix(data)')
    else
        return Matrix(Matrix(data[:,1:(ncol(data)-1)])')
    end
end 

function sig_gen(eigvals)
    n = length(eigvals)
    Q, ~ = qr(randn(n, n))
    D = Diagonal(eigvals) 
    return Q*D*Q'
end

end