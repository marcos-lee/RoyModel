
using Pkg
using Random
using Distributions
using LinearAlgebra
using Optim
using NLSolversBase
using Statistics



Random.seed!(1)

n = 10000

#set parameters

alpha0t = 1.2
alpha1t = .8

beta0t = 1.8
beta1t = .2


sigma0t = .8
sigma1t = 1.
sigma01t = .6


gammat = 1.1


X = rand(Normal(1,.4),n)
Z = rand(Normal(-1,.3),n)

mu = [0, 0]
sigma = [sigma0t sigma01t; sigma01t sigma1t]
epsilon = rand(MvNormal(mu, sigma),n)
epsilon = transpose(epsilon)

theta = [alpha1t, beta1t, gammat, alpha0t, beta0t, sigma0t, sigma01t]

Y0 = alpha0t .+ beta0t .* X .+ epsilon[:,1]
Y1 = alpha1t .+ beta1t .* X .+ gammat .* Z .+ epsilon[:,2]
S = Y1 .>= 0

Y0obs = Y0[ S]
Xobs = X[ S]
Zobs = Z[ S]


function mle1(theta, data, selection)
    x = data[:,2]
    y = data[:,1]
    z = data[:,3]
    s = selection
    x_sel = [ones(n) x z]
    delta1 = [theta[1], theta[2], theta[3]]
    x_main = [ones(n) x]
    delta0 = [theta[4], theta[5]]
    sigma0 = exp(theta[6])
    sigma01 = theta[7]
    x_sel*delta1
    epsilon = y .- x_main*delta0
    A = log.(1 .- cdf.(Normal(), x_sel*delta1))
    B = (x_sel*delta1 .+ (sigma01 ./ sigma0^2) .* (epsilon)) ./ sqrt(1 - sigma01^2 ./ sigma0^2)
    C = epsilon ./ sigma0
    D = log(sigma0)
    unsel = A
    sel = log.(cdf.(Normal(), B)) .+ log.(pdf.(Normal(), C)) .- D
    ll = - sum(unsel[ .!s]) - sum(sel[ s])
end


#let's get sensible starting parameters by doing biased OLS and probit
function probit(param, data)
    alphapr = param[1]
    betapr = param[2]
    gammapr = param[3]
    x = data[:,1]
    z = data[:,1]
    res = alphapr .+ data[:,1] .* betapr .+ data[:,2] .* gammapr
    q = 2 .* data[:,3] .- 1
    ll = cdf.(Normal(0,1),q .* res)
    LL = -sum(log.(ll))
end

function OLS(y,x)
    estimate = inv(transpose(x)*x)*(transpose(x)*y)
end

ols_biased = OLS(Y0obs, [ones(size(Xobs)) Xobs])
eps_ols = mean((Y0obs .- [ones(size(Xobs)) Xobs]*ols_biased).^2)
param0 = [.9, .3, 1.2]
datap = [X Z S]
estprobit = optimize(vars -> probit(vars, datap), param0)
param = Optim.minimizer(estprobit)

theta0 = [param[1], param[2], param[3], ols_biased[1], ols_biased[1], log(sqrt(eps_ols)), .1]

data = [Y0 X Z]

mini = optimize(vars -> mle1(vars, data, S), theta0)
theta = [alpha1t, beta1t, gammat, alpha0t, beta0t, log(sqrt(sigma0t)), sigma01t]

estdiff = Optim.minimizer(mini) - theta

exp(Optim.minimizer(mini)[6])^2
Optim.minimizer(mini)[6] = exp(Optim.minimizer(mini)[6])^2


B = 1000
samples = zeros(B,7)
for b = 1:B
    index = sample(1:n,n)
    data[:,2] = X[index,:]
    data[:,1] = Y0[index,:]
    data[:,3] = Z[index,:]
    selection = S[index]
    function wrapmle1(theta)
        return mle1(theta, data, selection)
    end
    samples[b,:] = optimize(wrapmle1, theta0).minimizer
end
samples[:,6] = exp.(samples[:,6]).^2
bootstrapSE = zeros(7)
for i = 1:7
    bootstrapSE[i] = std(samples[:,i])
end

Optim.minimizer(mini)
