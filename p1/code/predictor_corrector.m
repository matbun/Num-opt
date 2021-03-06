function [x, k] = predictor_corrector(A,b,c,x,lambda,s,maxiter,epsilon)
n = size(c,1);
m = size(A,1);

e = ones(n,1);

gamma = 0.99995;

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
    
    r_a = -(A*x-b);
    r_b = -(s+A'*lambda-c);
    r_c = -X*S*e;
    
    %solve the first system
    M = A*inv(S)*X*A'; %coefficients matrix to compute d_lambda_aff and d_lambda
    
    d_lambda_aff = M \ (r_a + A*inv(S)*X*r_b - A*inv(S)*r_c);
    d_s_aff = r_b - A'*d_lambda_aff;
    d_x_aff = inv(S)*r_c - inv(S)*X*d_s_aff;
    
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
    
    r_c = r_c - spdiags(d_x_aff,0,n,n)*spdiags(d_s_aff,0,n,n)*e + sigma*mu*e;
    
    %solve the second system
    d_lambda = M \ (r_a + A*inv(S)*X*r_b - A*inv(S)*r_c);
    d_s = r_b - A'*d_lambda;
    d_x = inv(S)*r_c - inv(S)*X*d_s;
    
    
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
    
    x = x + gamma*alfa_primal*d_x;
    lambda = lambda + gamma*alfa_dual*d_lambda;
    s = s + gamma*alfa_dual*d_s;
    
    mu = x'*s/n;
    if mu < epsilon*mu_0
        break
    end
end


end