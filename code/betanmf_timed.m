% This function decomposes input matrix by nonnegative matrix factorization
% (NMF) based on beta-divergence criterion and multiplicative update rules.
%
% The input nonnegative data matrix X will be decomposed as X = A*B
% approximately, where matrices A and B are often called "basis matrix" and
% "activation matrix," respectively. All the entries in X, A, and B must be
% nonnegative values. All the update rules implemented in this function are
% based on majorization-minimization algorithm.
%
% See also:
%
%   C. Fevotte, J. Idier, "Algorithms for nonnegative matrix factorization
%   with the beta-divergence."
%
%   M. Nakano, H. Kameoka, J. Le Roux, Y. Kitano, N. Ono, and S. Sagayama,
%   "Convergence-guaranteed multiplicative algorithms for nonnegative matrix
%   factorization with beta-divergence," Proc. MLSP, pp. 283-288, 2010.
%
%   http://d-kitamura.net
%
% Coded by D. Kitamura (d-kitamura@ieee.org) on 31 Jul, 2015 (ver1.0).
% Modified by D. Kitamura (d-kitamura@ieee.org) on 18 Nov, 2018 (ver1.1).
%
function [A, B, f, time_A, time_B, time_cost] = betanmf_timed (X, A, B, tol, maxiter, verbose)

  % Set zeros to small positive numbers to prevent numerical issues in
  % the updates below.
  [n p] = size(X);
  X     = max(X,eps);
  E     = ones(n,p);

  % Handle optional arguments.
  if nargin < 4
    tol = 1e-6;
  end
  if nargin < 5
    maxiter = 5000;
  end 
  if nargin < 6
    verbose = true;
  end
  
  n_steps_A = 7;
  n_steps_B = 7;
  time_A = zeros(maxiter, n_steps_A);
  time_B = zeros(maxiter, n_steps_B);
  time_cost = zeros(maxiter, 1);
  

  % Compute the value of the objective function at the initial estimate
  % of the solution.
  f    = zeros(maxiter + 1,1);
  f(1) = cost(X,A*B);
  if verbose
    fprintf('iter objective (cost fn) max.diff\n');
    fprintf('---- ------------------- --------\n');
  end

  % Repeat until the maximum number of iteration is reached, or until the
  % convergence criterion is met.
  for i = 1:maxiter
    A0     = A;
    B0     = B;
    [A B timeA timeB]  = update_timed(X,A,B,E);
    time_A(i,:) = timeA;
    time_B(i,:) = timeB;
    
    tic;
    f(i+1) = cost(X,A*B);
    time_cost(i) = toc;
    d      = max(max(max(abs(A - A0))),...
                 max(max(abs(B - B0))));
    if verbose
      fprintf('%4d %+0.12e %0.2e\n',i,f(i+1),d);
    end
    if d < tol
      break
    end
  end
  f = f(1:i+1);

  
function [A,B, timeA, timeB] = update_timed(X,A,B,E)
    [A, timeA] = update_timed_A(X, A, B, E);
    [B, timeB] = update_timed_B(X, A, B, E);

  
function [A,timeA] = update_timed_A(X, A, B, E)
    timeA = [];
%   A = A .* (((X ./ (A*B)) * B.') ./ (E * B.'));
%   A = max(A,eps);
%   B = B .* ((A.' * (X ./ (A*B))) ./ (A.' * E));
%   B = max(B,eps);
    % step 1
    tic;
    M = A*B;
    timeA = [timeA, [toc]];
    
    % step 2
    tic;
    M = X ./ M;
    timeA = [timeA, [toc]];
    
    % step 3
    tic;
    M = M * B';
    timeA = [timeA, [toc]];
    
    % step 4
    tic;
    D = E * B';
    timeA = [timeA, [toc]];
    
    % step 5
    tic;
    M = M ./ D;
    timeA = [timeA, [toc]];
    
    % step 6
    tic;
    A = A .* M;
    timeA = [timeA, [toc]];
    
    % step 7
    tic;
    A = max(A,eps);
    timeA = [timeA, [toc]];
    

function [B,timeB] = update_timed_B(X, A, B, E)
    timeB = [];
%   A = A .* (((X ./ (A*B)) * B.') ./ (E * B.'));
%   A = max(A,eps);
%   B = B .* ((A.' * (X ./ (A*B))) ./ (A.' * E));
%   B = max(B,eps);
    % step 1
    tic;
    M = A*B;
    timeB = [timeB, [toc]];
    
    % step 2
    tic;
    M = X ./ M;
    timeB = [timeB, [toc]];
    
    % step 3
    tic;
    M = A' * M;
    timeB = [timeB, [toc]];
    
    % step 4
    tic;
    D = A' * E;
    timeB = [timeB, [toc]];
    
    % step 5
    tic;
    M = M ./ D;
    timeB = [timeB, [toc]];
    
    % step 6
    tic;
    B = B .* M;
    timeB = [timeB, [toc]];
    
    % step 7
    tic;
    B = max(B,eps);
    timeB = [timeB, [toc]];
    
    





% This implements the multiplicative updates.
% function [A, B] = update (X, A, B, E)
%   A = A .* (((X ./ (A*B)) * B.') ./ (E * B.'));
%   A = max(A,eps);
%   B = B .* ((A.' * (X ./ (A*B))) ./ (A.' * E));
%   B = max(B,eps);
