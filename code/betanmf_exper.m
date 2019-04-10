% This function decomposes input matrix by nonnegative matrix factorization
% same as betanmf.m except that I save the some statistics of differences in A, B in the process to see how
% the update goes
%
function [A, B, f, maxdiffA, maxdiffB] = betanmf_exper (X, A, B, tol, maxiter, verbose, maxnum)

  % Set zeros to small positive numbers to prevent numerical issues in
  % the updates below.
  [n p] = size(X);
  X     = max(X,eps);
  E     = ones(n,p);
  
  %% save some As and Bs
  k = size(A,2);
  maxdiffA = zeros(maxiter, maxnum);
  maxdiffB = zeros(maxiter,maxnum);
  
  % Handle optional arguments.
  if nargin < 4
    tol = 1e-6;
  end
  if nargin < 5
    maxiter = 1000;
  end 
  if nargin < 6
    verbose = true;
  end

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
    [A B]  = update(X,A,B,E);
    diffA = abs(A - A0);
    diffB = abs(B - B0);
    maxdiffA(i,:) = maxk(diffA(:), maxnum)';
    maxdiffB(i,:) = maxk(diffB(:), maxnum)';
    
    f(i+1) = cost(X,A*B);
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

% This implements the multiplicative updates.
function [A, B] = update (X, A, B, E)
  A = A .* (((X ./ (A*B)) * B.') ./ (E * B.'));
  A = max(A,eps);
  B = B .* ((A.' * (X ./ (A*B))) ./ (A.' * E));
  B = max(B,eps);
