% TO DO: Update this script.
%
% Fit a non-megative matrix factorization to the GTEx data using the
% multiplication update rules; see betanmf.m for more information about the
% algorithm, and see fit_gtex_betanmf.sbatch for SLURM settings used on the
% RCC cluster.

% SCRIPT SETTINGS
% ---------------
% These variables specify the names of the input files.
datadir          = fullfile('..','output');
readcountsfile   = 'gtex.csv';
initfactorsfile  = 'gtex_factors_rough.csv';
initloadingsfile = 'gtex_loadings_rough.csv';

% These variables specify the names of the output files.
outdir          = fullfile('..','output');
factorsoutfile  = 'gtex_factors_admm.csv';
loadingsoutfile = 'gtex_loadings_admm.csv';

% Tuning parameter affecting convergence rate of ADMM algorithm.
r = 1;

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
fprintf('Fitting Poisson topic model using ADMM algorithm.\n')
tic;
[W H] = nmfadmm(counts,L0,F0',r);
timing = toc;
fprintf('Computation took %0.2f seconds.\n',timing);

% Convert the Poisson model parameters to the parameters for the
% multinomial model.
[F L] = poisson2multinom(H',W);

% Compute the multinomial likelihood for the nnmf solution.
f = loglikmultinom(counts,F,L);
fprintf('Multinomial likelihood at nnmf solution: %0.12f\n',f);

% WRITE NNMF RESULTS TO FILE
% --------------------------
fprintf('Writing results to file.\n');
factorsoutfile  = fullfile(outdir,factorsoutfile);
loadingsoutfile = fullfile(outdir,loadingsoutfile);
csvwrite(factorsoutfile,F);
csvwrite(loadingsoutfile,L);

% SESSION INFO
% ------------
ver
