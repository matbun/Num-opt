function [x_init,lambda_init,s_init] = starting_point(A,b,c)
e = ones(size(A,2),1);
x_tilde = A'*inv(A*A')*b;
lambda_tilde = inv(A*A')*A*c;
s_tilde = c - A'*lambda_tilde;

delta_x = max(-1.5*min(x_tilde),0);
delta_s = max(-1.5*min(s_tilde),0);

x_hat = x_tilde + delta_x*e;
s_hat = s_tilde + delta_s*e;

delta_x_hat = 0.5*x_hat'*s_hat/(e'*s_hat);
delta_s_hat = 0.5*x_hat'*s_hat/(e'*x_hat);

x_init = x_hat + delta_x_hat;
lambda_init = lambda_tilde;
s_init = s_hat + delta_s_hat;
end