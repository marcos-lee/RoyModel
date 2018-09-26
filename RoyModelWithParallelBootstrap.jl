
using Random
using Distributions
using LinearAlgebra
using Optim
using Statistics
using Distributed
#run only once, add processors for the parallelization
addprocs(3)


Random.seed!(1)

#set sample size
@everywhere n = 1000

#generate data and initial parameters of estimation
@everywhere include("DataGeneration.jl")


#obtain bootstrap standard errors

#B is number of bootstrap loops, b is the the number each processor will do
B = 100
b = 25
@everywhere include("BootstrapFunction.jl")
@elapsed samples_pmap = pmap(bootstrap, [b, b, b, b])
#819.391091792 for B=1000
#96.340405297 for B=100
samplesb = vcat(samples_pmap[1], samples_pmap[2], samples_pmap[3], samples_pmap[4])

#@elapsed bootstrap(B)
#2253.704890252 for B = 1000
#230.814269616 for B = 100
bootstrapSE1 = zeros(8)
for i = 1:8
    bootstrapSE1[i] = std(samplesb[:,i])
end
