clear 
close all
clc

a = 2e3;
n = 1e6;
m = 1;
c = ones(n,1);
c(1:2:end) = a;
A = ones(m,n);
b = ones(m,1);

e = ones(n,1);

[x, lambda, s] = starting_point(A,b,c);

I = speye(n);
S = spdiags(s,0,n,n);
X = spdiags(x,0,n,n);

AA = spalloc(2*n+m, 2*n+m, 5*n);
AA(1:m, 1:n) = A;
AA(m+1:m+n, n+1:m+n) = A';
AA(m+1:m+n, m+n+1:m+2*n) = I;
AA(m+n+1:m+2*n, 1:n) = S;
AA(m+n+1:m+2*n, m+n+1:m+2*n) = X;

%figure
%spy(AA)
%grid on

% numero di condizionameto
%condest(AA)

%AA positive definite? No if p > 0
%[~,p] = chol(AA)

%is AA singular? no
%sprank(AA)

r = [b - A*x; c-s-A'*lambda; -X*S*e];
d = AA\r;

r_c = s+A'*lambda-c;
r_b = A*x-b;
r_xs = X*S*e;

D = (inv(S)*X).^(1/2);
K = A*D.^2*A';
j = -r_b - A*X*inv(S)*r_c + A*inv(S)*r_xs;

d_lambda = K\j

d_s = -r_c -A'*d_lambda
d_x = -inv(S)*r_xs - X*inv(S)*d_s


for k = 0:maxiter
    
end
