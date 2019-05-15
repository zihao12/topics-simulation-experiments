% Fit a non-megative matrix factorization to the GTEx data using the
% multiplication update rules; see betanmf.m for more information about the
% algorithm, and see fit_gtex_betanmf.sbatch for SLURM settings used on the
% RCC cluster.

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
factorsoutfile  = join([dataname, '_factors_betanmf.csv']);
loadingsoutfile = join([dataname, '_loadings_betanmf.csv']);
erroutfile = join([dataname, '_error_betanmf.csv']);
timeAoutfile = join([dataname, '_timesA_betanmf.csv']);
timeBoutfile = join([dataname, '_timesB_betanmf.csv']);
timeCostoutfile = join([dataname, '_timesCost_betanmf.csv']);

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
[A, B, f, time_A, time_B, time_cost] = betanmf_timed(counts,L0,F0',1e-06,20);
timing = toc;
fprintf('Computation took %0.2f seconds.\n',timing);

% Convert the Poisson model parameters to the parameters for the
% multinomial model.
[F L] = poisson2multinom(B',A);

% Compute the multinomial likelihood for the nnmf solution.
f = loglikmultinom(counts,F,L);
fprintf('Multinomial likelihood at nnmf solution: %0.12f\n',f);

% WRITE NNMF RESULTS TO FILE
% --------------------------
fprintf('Writing results to file.\n');
% factorsoutfile  = fullfile(outdir,factorsoutfile);
% loadingsoutfile = fullfile(outdir,loadingsoutfile);
% erroutfile = fullfile(outdir,erroutfile);
timeAoutfile = fullfile(outdir,timeAoutfile);
timeBoutfile = fullfile(outdir,timeBoutfile);
timeCostoutfile = fullfile(outdir,timeCostoutfile);


% csvwrite(factorsoutfile,F);
% csvwrite(loadingsoutfile,L);
% csvwrite(erroutfile,err);
csvwrite(timeAoutfile,time_A);
csvwrite(timeBoutfile,time_B);
csvwrite(timeCostoutfile,time_cost);
% SESSION INFO
% ------------
ver
