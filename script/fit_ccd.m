% Fit a non-negative matrix factorization to the GTEx data using the cyclic
% co-ordinate descent (CCD) algorithm; see nmfccd.m for more information
% about the algorithm, and see fit_gtex_ccd.sbatch for Slurm settings
% used on the RCC cluster.

% SCRIPT SETTINGS
% ---------------
% These variables specify the names of the input files.
%% dataname = "...." is input in bacth file

datadir          = fullfile('../../topics-simulation-bigdata','output');
readcountsfile   = join([dataname, '.csv']);
initfactorsfile  = join([dataname, '_factors_rough.csv']);
initloadingsfile = join([dataname, '_loadings_rough.csv']);

% These variables specify the names of the output files.
outdir          = fullfile('../../topics-simulation-bigdata','output');
factorsoutfile  = join([dataname, '_factors_ccd.csv']);
loadingsoutfile = join([dataname, '_loadings_ccd.csv']);
erroutfile = join([dataname, '_error_ccd.csv']);


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

% PERFORM A FEW MULTIPLICATIVE UPDATES
% ------------------------------------
fprintf('Performing a few multiplicative updates using betanmf.\n')
tic;
[A0 B0 err] = betanmf(counts,L0,F0',1e-6,20);
timing = toc;
fprintf('Computation took %0.2f seconds.\n',timing);

% RUN NMF OPTIMIZATION METHOD
% ---------------------------
fprintf('Fitting Poisson topic model using CCD algorithm.\n')
tic;
[A B] = nmfccd(counts,A0,B0,1e-6,200);
timing = toc;
fprintf('Computation took %0.2f seconds.\n',timing);

% Convert the Poisson model parameters to the parameters for the
% multinomial model.
[F L] = poisson2multinom(B',A);

% Compute the multinomial likelihood for the CCD solution.
f = loglikmultinom(counts,F,L);
fprintf('Multinomial likelihood at CCD solution: %0.12f\n',f);

% WRITE CCD RESULTS TO FILE
% -------------------------
fprintf('Writing results to file.\n');
factorsoutfile  = fullfile(outdir,factorsoutfile);
loadingsoutfile = fullfile(outdir,loadingsoutfile);
erroutfile = fullfile(outdir,erroutfile);

csvwrite(factorsoutfile,B');
csvwrite(loadingsoutfile,A);
csvwrite(erroutfile,err);

% SESSION INFO
% ------------
ver
