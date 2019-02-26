% A short script illustrating the use of different algorithms for
% non-negative matrix factorization in a small data set.
addpath ../code

% SCRIPT PARAMETERS
% -----------------
% Number of factors ("topics").
K = 2;

% SET UP ENVIRONMENT
% ------------------
% Initialize the sequence of pseudorandom numbers.
rng(1);

% LOAD DATA
% ---------
% For this small demonstration, we use Fisher's iris data, stored as a 
% 150 x 4 matrix.
fprintf('Loading data.\n');
load fisheriris
X = meas;
clear meas species

% INITIALIZE ESTIMATES
% --------------------
% Initialize the estimates to random values.
n = size(X,1);
p = size(X,2);
A = rand(n,K);
B = rand(K,p);

% RUN MULTIPLICATIVE UPDATES
% --------------------------
% This is a very simple algorithm (originally proposed by Lee & Seung,
% 2001), but expected to take a long time to converge to a solution, and
% sometimes is not guaranteed to converge at all.
fprintf('Running multiplicative updates.\n');
[A1 B1 f1] = betanmf(X,A,B);

% RUN ADMM ALGORITHM
% ------------------
% This is the Alternating Direction Method of Multipliers (ADMM)
% algorithm developed by Sun & Fevotte (2014).
fprintf('Running ADMM algorithm.\n');
[A2 B2 f2] = nmfadmm(X,A,B,1);

% RUN CCD ALGORITHM
% -----------------
% This is the Cyclic Co-ordinate Descent (CCD) method described by Hsieh &
% Dhillon (2011).
fprintf('Running CCD updates.\n')
[A3 B3 f3] = nmfccd(X,A,B);

% SUMMARIZE RESULTS
% -----------------
fprintf('multiplicative: %0.12f\n',f1(end));
fprintf('ADMM algorithm: %0.12f\n',f2(end));
fprintf('CCD algorithm:  %0.12f\n',f3(end));