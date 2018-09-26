using Random
using Distributions
using LinearAlgebra
using Optim
using Statistics
using Distributed

function mle(theta,data)
    y0 = data[:,1]
    y1 = data[:,2]
    z = data[:,4]
    s = data[:,5]
    x = data[:,3]
    alpha0 = theta[1]
    beta0 = theta[2]
    alpha1 = theta[3]
    beta1 = theta[4]
    gamma = theta[5]
    sigma0 = exp(theta[6])
    sigma1 = exp(theta[7])
    sigma01 = theta[8]
    epsilon0 = y0 .- alpha0 .- x .* beta0
    epsilon1 = y1 .- alpha1 .- x .* beta1
    sel0 = alpha1 .- alpha0 .+ x .* beta1 .- x .* beta0 .- z .* gamma + (sigma01 - sigma0^2)/sigma0^2 .* epsilon0
    sel1 = alpha1 .- alpha0 .+ x .* beta1 .- x .* beta0 .- z .* gamma + (sigma1^2 - sigma01)/sigma1^2 .* epsilon1
    probitstd0 = sqrt(sigma1^2 + sigma0^2 - 2 * sigma01 - (sigma01 - sigma0^2)^2/sigma0^2)
    probitstd1 = sqrt(sigma1^2 + sigma0^2 - 2 * sigma01 - (sigma1^2 - sigma01)^2/sigma1^2)
    pdfeval0 = (y0 .- alpha0 .- x .* beta0) ./ sigma0
    pdfeval1 = (y1 .- alpha1 .- x .* beta1) ./ sigma1
    selection0 = log.(1 .- cdf.(Normal(), (sel0) ./ probitstd0)) .+ log.(pdf.(Normal(), pdfeval0)) .- log(sigma0)
    selection1 = log.(cdf.(Normal(), (sel1) ./ probitstd1)) .+ log.(pdf.(Normal(), pdfeval1)) .- log(sigma1)
    ll = -(sum(selection0[ s .== 0]) + sum(selection1[ s .== 1]))
end

function bootstrap(B)
println("hi")
    samples = zeros(B,8)
    for b = 1:B
        index = sample(1:n,n)
        data[:,1] = Y0[index,:]
        data[:,2] = Y1[index,:]
        data[:,3] = X[index,:]
        data[:,4] = Z[index,:]
        data[:,5] = S[index,:]
        function wrapmle(theta)
            return mle(theta, data)
        end
        samples[b,:] = optimize(wrapmle, theta0).minimizer
    end
    samples[:,6] = exp.(samples[:,6]).^2
    samples[:,7] = exp.(samples[:,7]).^2
    println("bye")
    return samples
end
