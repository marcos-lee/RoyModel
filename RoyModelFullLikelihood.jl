

using Pkg
using Random
using Distributions
using LinearAlgebra
using Optim
using NLSolversBase
using Statistics



Random.seed!(1)

n = 1000

#set parameters

alpha0t = 1.2
alpha1t = 2

beta0t = 3.5
beta1t = 4

sigma0t = 2.
sigma1t = 3.
sigma01t = 1.2


gammat = 1.1

theta = [alpha0t beta0t alpha1t beta1t gammat sigma0t sigma1t sigma01t]
X = rand(Normal(8,3),n)
Z = rand(Normal(4,1),n)

mu = [0, 0]
sigma = [sigma0t sigma01t; sigma01t sigma1t]
epsilon = rand(MvNormal(mu, sigma),n)
epsilon = transpose(epsilon)

Y0 = alpha0t .+ beta0t .* X .+ epsilon[:,1]
Y1 = alpha1t .+ beta1t .* X .+ epsilon[:,2]
S = Y1 .- Z .* gammat .>= Y0


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


#let's get sensible initial guesses from biased OLSs

function OLS(y,x)
    estimate = inv(transpose(x)*x)*(transpose(x)*y)
end

data0 = [ones(n) X][ .!S, :]
data1 = [ones(n) X][ S, :]

ols0 = OLS(Y0[ .!S], data0)
ols1 = OLS(Y1[ S], data1)

eps0 = mean((Y0[ .!S] .- data0 * ols0).^2)
eps1 = mean((Y1[ S] .- data1 * ols1).^2)


data = [Y0 Y1 X Z S]
theta0 = [ols0[1] ols0[2] ols1[1] ols1[2] 1. eps0 eps1 1.]

mini = optimize(vars -> mle(vars, data), theta0)

Optim.minimizer(mini)[6] = exp(Optim.minimizer(mini)[6])^2
Optim.minimizer(mini)[7] = exp(Optim.minimizer(mini)[7])^2

estdiff = Optim.minimizer(mini) - theta


#obtain bootstrap standard errors
data = [Y0 Y1 X Z S]

B = 100
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

bootstrapSE = zeros(8)
for i = 1:8
    bootstrapSE[i] = std(samples[:,i])
end
