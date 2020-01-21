function [x, k] = predictor_corrector_lu(A,b,c,x,lambda,s,maxiter,epsilon)
n = size(c,1);
m = size(A,1);

e = ones(n,1);

mu = x'*s/n;
mu_0 = mu;

AA = spalloc(2*n+m, 2*n+m, 5*n);
AA(1:m, 1:n) = A;
AA(m+1:m+n, n+1:m+n) = A';
AA(m+1:m+n, m+n+1:m+2*n) = speye(n);

for k = 0:maxiter
    X = spdiags(x,0,n,n);
    S = spdiags(s,0,n,n);
    AA(m+n+1:m+2*n, 1:n) = S;
    AA(m+n+1:m+2*n, m+n+1:m+2*n) = X;
    
    r_c = s+A'*lambda-c;
    r_b = A*x-b;
    r_xs = X*S*e;
    
    r = [-r_b; -r_c; -r_xs];
    [L,U] = lu(AA);
    y = L\r;
    r = U\y;

    d_lambda_aff = r(n+1);
    d_s_aff = r(n+2:end);
    d_x_aff = r(1:n);
    
    if isempty(d_x_aff(d_x_aff<0))
        alfa_aff_primal = 1;
    else
        alfa_aff_primal = min(1,min(-x(d_x_aff<0)./d_x_aff(d_x_aff<0)));
    end
    
    if isempty(d_s_aff(d_s_aff<0))
        alfa_aff_dual = 1;
    else
        alfa_aff_dual = min(1,min(-s(d_s_aff<0)./d_s_aff(d_s_aff<0)));
    end
    
    mu_aff = (x + alfa_aff_primal*d_x_aff)'*(s + alfa_aff_dual*d_s_aff)/n;
    sigma = (mu_aff/mu)^3;
    
    r_xs = r_xs + spdiags(d_x_aff,0,n,n)*spdiags(d_s_aff,0,n,n)*e - sigma*mu*e;
    
    %solve the second system
    r = [-r_b; -r_c; -r_xs];
    y = L\r;
    r = U\y;
    
    d_lambda = r(n+1);
    d_s = r(n+2:end);
    d_x = r(1:n);
    
    
    alfa_primal_max = min(-x(d_x_aff<0)./d_x_aff(d_x_aff<0));
    alfa_dual_max = min(-s(d_s_aff<0)./d_s_aff(d_s_aff<0));
    
    eta = max(1-mu,0.9);
    
    if isempty(alfa_primal_max)
        alfa_primal = 1;
    else
        alfa_primal = min(1,eta*alfa_primal_max);
    end
    
    if isempty(alfa_dual_max)
        alfa_dual = 1;
    else
        alfa_dual = min(1,eta*alfa_dual_max);
    end
    
    x = x + alfa_primal*d_x;
    lambda = lambda + alfa_dual*d_lambda;
    s = s + alfa_dual*d_s;
    
    mu = x'*s/n;
    if mu < epsilon*mu_0
        break
    end
end


end