% Implements NMF algorithm described in:
%
%   D.L. Sun and C. Fevotte. "Alternating direction method of multipliers
%   for non-negative matrix factorization with the beta divergence." ICASSP
%   2014.
%
% Original source: 
%
%   https://web.calpoly.edu/~dsun09/admm.html
%
function [W, H, f] = nmfadmm (V, W, H, r, tol, maxiter, verbose)
    
  % Determine problem dimensions.
  [m n] = size(V);
  k     = size(W,2);

  % Handle optional arguments.
  if nargin < 4
    r = 1; 
  end
  if nargin < 5
    tol = 1e-6;
  end
  if nargin < 6
    maxiter = 1000;
  end
  if nargin < 7
    verbose = true;
  end
    
  % Initializations for other variables.
  X  = W*H;
  Wp = W;
  Hp = H;
  aX = zeros(size(X));
  aW = zeros(size(W));
  aH = zeros(size(H));

  % Compute the value of the objective function at the initial estimate
  % of the solution.
  f    = zeros(maxiter + 1,1);
  f(1) = cost(X,W*H);
  if verbose
    fprintf('iter objective (cost fn) max.diff\n');
    fprintf('---- ------------------- --------\n');
  end

  % Repeat until the maximum number of iteration is reached, or until the
  % convergence criterion is met.
  for i = 1:maxiter
    W0 = Wp;
    H0 = Hp;

    % Update for H.
    H = (W'*W + eye(k)) \ (W'*X + Hp + (W'*aX - aH)/r);
        
    % Update for W.
    P = H*H' + eye(k);
    Q = H*X' + Wp' + (H*aX' - aW')/r;
    W = (P \ Q)';
        
    % Update for X.
    Xap = W*H;
    b   = r*Xap - aX - 1;
    X   = (b + sqrt(b.^2 + 4*r*V))/(2*r);

    % Update for H+ and W+.
    Hp = max(H + aH/r,0);
    Wp = max(W + aW/r,0);
        
    % Update for the dual variables.
    aX = aX + r * (X - Xap);
    aH = aH + r * (H - Hp);
    aW = aW + r * (W - Wp);

    % Compute the value of the objective at the current solution estimate, and
    % check the convergence criterion.
    f(i+1) = cost(X,Wp*Hp);
    d      = max(max(max(abs(Wp - W0))),...
                 max(max(abs(Hp - H0))));
    if verbose
      fprintf('%4d %+0.12e %0.2e\n',i,f(i+1),d);
    end
    if (i > 1) & (d < tol)
      break
    end
  end

  W = Wp;
  H = Hp; 
  f = f(1:i+1);
