// See nmfccd.m for an overview of what this code does.
#include <string.h>
#include <math.h>
#include <mex.h>

#define max(a,b) ((a) > (b) ? (a) : (b))

// FUNCTION DECLARATIONS
// ---------------------
void   copy   (const double* source, double* dest, size_t n);
double cost   (int n, int m, const  double* V, const double* WH, double e);
double update (int m, int k, double* wt, double* wht, const double* vt, 
	       double* H, double e);
int    ccd    (int n, int m, int k, const double* V, double* W, double* H, 
	       double* WH, double* f, double* vt, double* wht, double e, 
	       double tol, int maxiter, int verbose);

// FUNCTION DEFINITIONS
// --------------------
void mexFunction (int nlhs, mxArray* plhs[], 
		  int nrhs, const mxArray* prhs[]) {

  // GET INPUTS
  // ----------
  const double* V  = mxGetPr(prhs[0]);
  const double* W0 = mxGetPr(prhs[1]);
  const double* H0 = mxGetPr(prhs[2]);
  double*       WH = mxGetPr(prhs[3]);

  const double e       = mxGetScalar(prhs[4]);
  const double tol     = mxGetScalar(prhs[5]);
  const int    maxiter = (int) mxGetScalar(prhs[6]);
  const int    verbose = (int) mxGetScalar(prhs[7]);

  // Get the dimensions of the data matrix (n x m), and the number of
  // factors in the non-negative matrix factorization (k).
  const size_t n = mxGetM(prhs[0]);
  const size_t m = mxGetN(prhs[0]);
  const size_t k = mxGetM(prhs[1]);

  // INITIALIZE OUTPUTS
  // ------------------
  plhs[0] = mxCreateDoubleMatrix(k,n,mxREAL);
  plhs[1] = mxCreateDoubleMatrix(k,m,mxREAL);
  plhs[2] = mxCreateDoubleMatrix(1,maxiter,mxREAL);
  plhs[3] = mxCreateDoubleScalar(mxREAL);
  double* W = mxGetPr(plhs[0]);
  double* H = mxGetPr(plhs[1]);
  double* f = mxGetPr(plhs[2]);
  double* numiter = mxGetPr(plhs[3]);
  copy(W0,W,n*k);
  copy(H0,H,m*k);

  // RUN THE CCD UPDATES
  // -------------------
  if (verbose) {
    mexPrintf("iter objective (cost fn) max.diff\n");
    mexPrintf("---- ------------------- --------\n");
  }
  double* wht = malloc(sizeof(double) * m); 
  double* vt  = malloc(sizeof(double) * m);
  *numiter = (double) ccd(n,m,k,V,W,H,WH,f,vt,wht,e,tol,maxiter,verbose);
  free(wht);
  free(vt);
}

// Copy entries of one array of doubles to another.
void copy (const double* source, double* dest, size_t n) {
  memcpy(dest,source,sizeof(double)*n);
}

// Compute the value of the objective function up to a normalizing
// constant. 
double cost (int n, int m, const double* V, const double* WH, double e) {
  double t = n * m;
  double y = 0;
  for (int i = 0; i < t; i++)
    y += WH[i] - V[i] * log(WH[i] + e);
  if (isnan(y))
    y = INFINITY;
  return y;
}

// Implements the co-ordinate descent updates for the factors (H) and
// loadings (W).
double update (int m, int k, double* wt, double* wht, const double* vt, 
	       double* H, double e) {
  int i, j, hi;
  double d, g, h, t, w0, w1;
  double dmax = 0;

  for (i = 0; i < k; i++) {
    g = 0;
    h = 0;
    for (j = 0, hi = i; j < m; j++, hi += k) {
      t  = vt[j]/(wht[j] + e);
      g += H[hi]*(1 - t);
      h += H[hi]*H[hi]*t/(wht[j] + e);
    }
    w0 = wt[i];
    w1 = wt[i] - g/h + e;
    if (w1 < e)
      w1 = e;
    d     = w1 - w0;
    dmax  = max(dmax,fabs(d));
    wt[i] = w1;
    for (j = 0; j < m; j++)
      wht[j] += d * H[j*k + i];
  }

  return dmax;
}

// Implements the cyclic co-ordinate descent updates described in
// Hsieh & Dhillon (2011).
int ccd (int n, int m, int k, const double* V, double* W, double* H, 
	 double* WH, double* f, double* vt, double* wht, double e, 
	 double tol, int maxiter, int verbose) {
  int iter, i, j;
  double d, dh, dw;

  // Repeat until the maximum number of iterations is reached, or
  // until the convergence criterion is met.
  for (iter = 0; iter < maxiter; iter++) {

    // Update W.
    dw = 0;
    for (i = 0; i < n; i++) {
      for (j = 0; j < m; j++) {
        wht[j] = WH[j*n + i];
        vt[j]  = V[j*n + i];
      }
      d  = update(m,k,W + i*k,wht,vt,H,e);
      dw = max(dw,d);
      for (j = 0; j < m; j++)
        WH[j*n + i] = wht[j];
    }

    // Update H.
    dh = 0;
    for (i = 0; i < m; i++) {
      d  = update(n,k,H + i*k,WH + i*n,V + i*n,W,e);
      dh = max(dh,d);
    }

    // Compute the value of the objective at the current iterate, and
    // check convergence.
    f[iter] = cost(n,m,V,WH,e);
    if (verbose) {
      mexPrintf("%4d %+0.12e %0.2e",iter + 1,f[iter],max(dh,dw));
      mexEvalString("disp(' ')");
    }
    if ((dh < tol) && (dw < tol))
      return iter + 1;
  }

  return maxiter;
}
