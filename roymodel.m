% set DGP
% Parameter space: alpha_0, alpha_1, beta_0, beta_1, variance matrix, gamma
clear;
rng(123);

alpha0 = 1.2;
alpha1 = 2;

beta0 = 3.5;
beta1 = 4;

sigma0 = 2;
sigma1 = 3;
sigma01 = 1.2;

gamma = 1.1;

% draw X and Z
n = 100000;

X = 3*randn(n,1) + 8;
Z = randn(n,1) + 4;

% draw error terms of wage equations from multivariate normal distribution
mu = [0 0];
sigma = [sigma0 sigma01; sigma01 sigma1];
eps = mvnrnd(mu, sigma, n);

% create Y0, Y1 and S from theoretical model

Y0 = alpha0 + beta0 * X + eps(:,1);
Y1 = alpha1 + beta1 * X + eps(:,2);
S = Y1 - Z * gamma >= Y0;


% let's estimate the first stage manually.
theta0 = [1,1,1]; %this is an initial guess for the optimizer
data = [X Z S]; % collects the data in a matrix
options=optimset('Display','off','MaxIter',10000,'TolX',10^-30,'TolFun',10^-30); % set of options for the optimization
theta = fminunc('probit', theta0, options, data); % fminunc is a minimizer of unconstrained multivariate function
%in matlab, functions have to be in a different file (but in the same
%directory path). I wrote the log likelihood of the probit model in the
%file probit.m

%below we collect the estimates from the first stage.
%Note that they are not the final estimates, but scaled by var(eps1 - eps0)
alphadiffp = theta(1);
betadiffp = theta(2);
gammap = theta(3);


%Now let's estimate the second stage, the OLS with selection correction.
% we have to estimate 2 OLS, one when S=0, another when S=1.

%data0 collects the variables when S=0
data0 = [Y0 X Z S];
%this is conditioning data0 matrix when S=0 (data0(:,4)=0)
data0 = data0(data0(:,4)==0, 1:3);

%analogous for S=1
data1 = [Y1 X Z S];
data1 = data1(data1(:,4)==1, 1:3);

%eval0 and eval1 is the term that we evaluate the mills ratio
eval0 = alphadiffp + data0(:,2).*betadiffp + data0(:,3).*gammap;
mills0 = normpdf(eval0)./normcdf(eval0);

eval1 = alphadiffp + data1(:,2).*betadiffp + data1(:,3).*gammap;
mills1 = normpdf(eval1)./(1-normcdf(eval1));

%this is the second stage OLS selection
olsX0 = [ones(size(data0(:,1))) data0(:,2) mills0];
olsX1 = [ones(size(data1(:,1))) data1(:,2) mills1];
b0 = regress(data0(:,1), olsX0);
b1 = regress(data1(:,1), olsX1);

%from this stage, we can recover the estimates for alpha0, alpha1, beta0,
%beta1 directly, and gamma and var(eps1-eps0) indirectly
alpha0hat = b0(1);
alpha1hat = b1(1);

beta0hat = b0(2);
beta1hat = b1(2);

vardiff = (alpha0hat-alpha1hat) / alphadiffp; %or alpha1hat / alpha1p. this is var(eps1-eps0)

gammahat = gammap*vardiff;


%captures the residuals for each OLS
resid0 = data0(:,1) - olsX0*b0;
resid1 = data1(:,1) - olsX1*b1;

%finally, the third step recovers the variance matrix
sigma0hat = mean(resid0.^2) - b0(3)^2 .* mean(- mills0.*(eval0) -  mills0.^2);
sigma1hat = mean(resid1.^2) - b1(3)^2 .* mean(mills1.*(eval1) - mills1.^2);

sigma01hat = (sigma0hat + sigma1hat - vardiff^2)/2;

%instead of doing 'plug-in' estimator, do OLS like Flavio said in class

varb0 = regress(resid0.^2, [ones(size(data0(:,1))) -mills0.*(eval0)-mills0.^2]);

varb1 = regress(resid1.^2, [ones(size(data1(:,1))) mills1.*(eval1)-mills1.^2]);

sigma0hatAlt = varb0(1);
sigma1hatAlt = varb1(1);

sigma01hatAlt = (sigma0hatAlt + sigma1hatAlt - vardiff^2)/2;
%so we can compare how accurate our estimates are, I collect them and the true
%parameters from DGP

theta_true = [alpha0, alpha1, beta0, beta1, gamma, sigma0, sigma1, sigma01];
theta_est = [alpha0hat, alpha1hat, beta0hat, beta1hat, gammahat, sigma0hat, sigma1hat, sigma01hat];
