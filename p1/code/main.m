clear 
close all
clc

m = 1;

maxiter = 1000;
epsilon = 1e-16;

n_iter = zeros(2,3);
elapsed_t = zeros(2,3);

i = 1;
for n = [1e4]
    j = 1;
    for a = [2 20 2e2 2e3]
        
        c = ones(n,1);
        c(1:2:end) = a;
        A = ones(m,n);
        b = ones(m,1);

        %[x, lambda, s] = starting_point(A,b,c);
        x = ones(n,1);
        s = ones(n,1);
        lambda = 1;
        
        tic
        [~, iter] = predictor_corrector(A,b,c,x,lambda,s,maxiter,epsilon);
        
        elapsed_t(i,j) = toc;
        n_iter(i,j) = iter;
        
        j = j+1;
    end 
    i = i+1;
end


elapsed_t
n_iter






