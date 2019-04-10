% A short script illustrating the use of different algorithms for
% non-negative matrix factorization in a small data set.
%addpath ../code

% SCRIPT PARAMETERS
% -----------------
% Number of factors ("topics").
K = 3;
n = 100;
p = 500;
MAXITER = 1000;
maxtop = 3;
tol = 1e-10;
verbose = 1;
% SET UP ENVIRONMENT
% ------------------
% Initialize the sequence of pseudorandom numbers.
rng(1);

% GENERATE DATA
% ---------
fprintf('Generating data.\n');
A_true = rand(n,K);
A_true = exp(A_true);
B_true = rand(K,p);
B_true = exp(B_true);
Lam = A_true*B_true;
X = poissrnd(Lam, n, p);
f0 = cost(X, Lam);


% INITIALIZE ESTIMATES
% --------------------
% Initialize the estimates to random values.
A = rand(n,K);
B = rand(K,p);

% RUN MULTIPLICATIVE UPDATES
% --------------------------
% This is a very simple algorithm (originally proposed by Lee & Seung,
% 2001), but expected to take a long time to converge to a solution, and
% sometimes is not guaranteed to converge at all.
fprintf('Running multiplicative updates.\n');
[A1 B1 f1 maxdiffA maxdiffB] = betanmf_exper(X,A,B,tol, MAXITER, verbose, maxtop);

% % RUN ADMM ALGORITHM
% % ------------------
% % This is the Alternating Direction Method of Multipliers (ADMM)
% % algorithm developed by Sun & Fevotte (2014).
% fprintf('Running ADMM algorithm.\n');
% [A2 B2 f2] = nmfadmm(X,A,B,1);
% 
% % RUN CCD ALGORITHM
% % -----------------
% % This is the Cyclic Co-ordinate Descent (CCD) method described by Hsieh &
% % Dhillon (2011).
% fprintf('Running CCD updates.\n')
% [A3 B3 f3] = nmfccd(X,A,B);

% SUMMARIZE RESULTS
% -----------------
fprintf('multiplicative: %0.12f\n',f1(end));
% fprintf('ADMM algorithm: %0.12f\n',f2(end));
% fprintf('CCD algorithm:  %0.12f\n',f3(end));