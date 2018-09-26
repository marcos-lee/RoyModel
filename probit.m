function LL = probit(theta, data)
    alphadiffp = theta(1);
    betadiffp = theta(2);
    gammap = theta(3);
    eval = alphadiffp + data(:,1).*betadiffp + data(:,2).*gammap;
    q = -2*data(:,3)+1;
    ll = normcdf(q.*eval);
    LL = -sum(log(ll));
end