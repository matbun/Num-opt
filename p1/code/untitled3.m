n = 1e4;
a = 2000;
m = 1;

c = ones(n,1);
c(1:2:end) = a;
A = ones(m,n);
b = ones(m,1);

%[x, lambda, s] = starting_point(A,b,c);
x = 100*ones(n,1);
s = 100*ones(n,1);
lambda = 100;

e = ones(n,1);

AA = spalloc(2*n+m, 2*n+m, 5*n);
AA(1:m, 1:n) = A;
AA(m+1:m+n, n+1:m+n) = A';
AA(m+1:m+n, m+n+1:m+2*n) = speye(n);
X = spdiags(x,0,n,n);
S = spdiags(s,0,n,n);
AA(m+n+1:m+2*n, 1:n) = S;
AA(m+n+1:m+2*n, m+n+1:m+2*n) = X;

[L,U,P,Q] = lu(AA);

figure
spy(L)
figure
spy(U)


