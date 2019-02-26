% Fit a non-megative matrix factorization to the GTEx data using the
% multiplication update rules; see betanmf.m for more information about the
% algorithm, and see fit_gtex_betanmf.sbatch for SLURM settings used on the
% RCC cluster.

% SCRIPT SETTINGS
% ---------------
% These variables specify the names of the input files.
datadir          = fullfile('../../topics-simulation-bigdata','output');
readcountsfile   = 'gtex_simulation_nnlm.csv';
initfactorsfile  = 'gtex_simulation_rough_factors.csv';
initloadingsfile = 'gtex_simulation_rough_loadings.csv';

% readcountsfile   = 'test.csv';
% initfactorsfile  = 'test_factors.csv';
% initloadingsfile = 'test_loadings.csv';


% These variables specify the names of the output files.
outdir          = fullfile('../../topics-simulation-bigdata','output');
factorsoutfile  = 'gtex_simulation_factors_betanmf.csv';
loadingsoutfile = 'gtex_simulation_loadings_betanmf.csv';

% SET UP ENVIRONMENT
% ------------------
addpath ../code

% LOAD GTEX DATA
% --------------
fprintf('Loading GTEx data.\n');
readcountsfile = fullfile(datadir,readcountsfile);
counts = csvread(readcountsfile);
fprintf('Loaded %d x %d count matrix.\n',size(counts,1),size(counts,2));

% LOAD INITIAL ESTIMATES
% ----------------------
fprintf('Loading initial estimates of factors and loadings.\n');
initfactorsfile  = fullfile(datadir,initfactorsfile);
initloadingsfile = fullfile(datadir,initloadingsfile);
F0               = csvread(initfactorsfile);
L0               = csvread(initloadingsfile);
fprintf('Loaded %d x %d factors matrix, ',size(F0,1),size(F0,2));
fprintf('and %d x %d loadings matrix.\n',size(L0,1),size(L0,2));

% RUN NMF OPTIMIZATION METHOD
% ---------------------------
fprintf('Fitting Poisson topic model using betanmf.\n')
tic;
[A B] = betanmf(counts,L0,F0');
timing = toc;
fprintf('Computation took %0.2f seconds.\n',timing);

% Compute the poisson likelihood for the nnmf solution.



% Convert the Poisson model parameters to the parameters for the
% multinomial model.
% [F L] = poisson2multinom(B',A);
F = B'
L = A

fprintf('Writing results to file.\n');
factorsoutfile  = fullfile(outdir,factorsoutfile);
loadingsoutfile = fullfile(outdir,loadingsoutfile);
csvwrite(factorsoutfile,F);
csvwrite(loadingsoutfile,L);



% Compute the multinomial likelihood for the nnmf solution.
fprintf("Compute loglikelihood of rough fit\n")
[type,multinom_ll, pois_ll] = compute_loglik(counts',F,L')

fprintf("method type: %s\n", type) 
fprintf("poisson_ll  : %s\n", pois_ll) 
fprintf("multinom_ll : %s\n", multinom_ll) 

% f = loglikmultinom(counts,F,L);
% fprintf('Multinomial likelihood at nnmf solution: %0.12f\n',f);

% WRITE NNMF RESULTS TO FILE
% --------------------------

% SESSION INFO
% ------------
ver
