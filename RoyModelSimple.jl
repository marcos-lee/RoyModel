
using Pkg
using Random
using Distributions
using LinearAlgebra
using Optim


Random.seed!(1)

n = 10000

#set parameters

alpha0 = 1.2
alpha1 = 2

beta0 = 3.5
beta1 = 4

sigma0 = 2.
sigma1 = 3.
sigma01 = 1.2


gamma = 1.1


X = rand(Normal(8,3),n)
Z = rand(Normal(4,1),n)

mu = [0, 0]
sigma = [sigma0 sigma01; sigma01 sigma1]
epsilon = rand(MvNormal(mu, sigma),n)
epsilon = transpose(epsilon)

Y0 = alpha0 .+ beta0 .* X .+ epsilon[:,1]
Y1 = alpha1 .+ beta1 .* X .+ epsilon[:,2]
S = Y1 .- Z .* gamma .>= Y0

function probit(theta, data)
    alphadiffp = theta[1]
    betadiffp = theta[2]
    gammap = theta[3]
    res = alphadiffp .+ data[:,1] .* betadiffp .+ data[:,2] .* gammap
    q = -2 .* data[:,3] .+ 1
    ll = cdf.(Normal(0,1),q .* res)
    LL = -sum(log.(ll))
end


theta0 = [1.2, 1.2, 1.2]
data = [X Z S]

mini = optimize(vars -> probit(vars, data), theta0)

alphadiffp = Optim.minimizer(mini)[1]
betadiffp = Optim.minimizer(mini)[2]
gammap = Optim.minimizer(mini)[3]

data0 = [Y0 ones(n) X Z][ .!S, :]
data1 = [Y1 ones(n) X Z][ S, :]

eval0 = alphadiffp .+ data0[:,3] .* betadiffp .+ data0[:,4] .* gammap
mills0 = pdf.(Normal(),eval0)./cdf.(Normal(),eval0)

eval1 = alphadiffp .+ data1[:,3] .* betadiffp .+ data1[:,4] .* gammap
mills1 = pdf.(Normal(),eval1)./(1 .- cdf.(Normal(),eval1))


function OLS(y,x)
    estimate = inv(transpose(x)*x)*(transpose(x)*y)
end

#there is something wrong with data for OLS1
#alpha1 is off by 0.2. It is not the OLS function.

b0 = OLS(data0[:,1], [data0[:,2:3] mills0]);
b1 = OLS(data1[:,1], [data1[:,2:3] mills1]);

teste0 = [data0[:,1:3] mills0]
teste1 = [data1[:,1:3] mills1]

#from this stage, we can recover the estimates for alpha0, alpha1, beta0,
#beta1 directly, and gamma and var(eps1-eps0) indirectly
alpha0hat = b0[1];
alpha1hat = b1[1];

beta0hat = b0[2];
beta1hat = b1[2];

vardiff = (alpha0hat-alpha1hat) / alphadiffp; #or alpha1hat / alpha1p. this is var(eps1-eps0)

gammahat = gammap*vardiff;


#captures the residuals for each OLS
resid0 = data0[:,1] - [data0[:,2:3] mills0]*b0;
resid1 = data1[:,1] - [data1[:,2:3] mills1]*b1;

#finally, the third step recovers the variance matrix
sigma0hat = mean(resid0.^2) .- b0[3]^2 .* mean(- mills0 .* eval0 .-  mills0.^2);
sigma1hat = mean(resid1.^2) - b1[3]^2 .* mean(mills1 .* eval1 .- mills1.^2);

sigma01hat = (sigma0hat + sigma1hat - vardiff^2)/2;
