
using Random
using Distributions
using LinearAlgebra
using Optim
using Statistics
using Distributed
#set true parameters

alpha0t = 1.2
alpha1t = 2
beta0t = 3.5
beta1t = 4
sigma0t = 2.
sigma1t = 3.
sigma01t = 1.2
gammat = 1.1

@everywhere theta = [alpha0t beta0t alpha1t beta1t gammat sigma0t sigma1t sigma01t]
X = rand(Normal(8,3),n)
Z = rand(Normal(4,1),n)

mu = [0, 0]
sigma = [sigma0t sigma01t; sigma01t sigma1t]
epsilon = rand(MvNormal(mu, sigma),n)
epsilon = transpose(epsilon)

Y0 = alpha0t .+ beta0t .* X .+ epsilon[:,1]
Y1 = alpha1t .+ beta1t .* X .+ epsilon[:,2]
S = Y1 .- Z .* gammat .>= Y0

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
