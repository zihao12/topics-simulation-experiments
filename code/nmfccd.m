% See demo_nmf.m for an example of how to use this function.
% 
% Steps I took to compile MEX file on midway2:
%
%   module load gcc/6.3.0
%   module load matlab/2018b
%   mex nmfccd_helper.cpp
%
function [A, B, f] = nmfccd (X, A, B, tol, maxiter, verbose) 

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

  % Run the cyclic co-ordinate descent (CCD) algorithm.
  %
  % Note that the matrix "AB" is modified by "nmfccd_helper" (this is not
  % particularly good practice, but it makes implementation much simpler).
  %
  AB = A * B;
  A = A';
  [A B f n] = nmfccd_helper(X,A,B,AB,eps,tol,maxiter,verbose);
  A = A';
  f = f(1:n);
