#include <math.h>
#include "mex.h"
#include "common.h"
#include <iostream>

void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    int n, p;
    if (nlhs != 2) {
        mexErrMsgTxt("Output should countain 2 argument: w, lambda");
    }
    
    if (nrhs < 10) {
        mexErrMsgTxt("Input should contain 10 arguments: X, y, method, gamma, lambda_min_ratio, nlambda, normalize, threshold, maxiter, nthread");
    }
    double *X = mxGetPr(prhs[0]);
    double *y = mxGetPr(prhs[1]);
    int method = (int) mxGetScalar(prhs[2]); // 1: MCP; 2: SCAD; 3: lasso
    double gamma = mxGetScalar(prhs[3]);
    double lambda_min_ratio = mxGetScalar(prhs[4]);
    int nlambda = (int) mxGetScalar(prhs[5]);
    int normalize = (int) mxGetScalar(prhs[6]);
    double threshold = mxGetScalar(prhs[7]);
    int maxiter = (int) mxGetScalar(prhs[8]);
    int nthread = (int) mxGetScalar(prhs[9]);
    
    n = (int) mxGetM(prhs[0]);
    p = (int) mxGetN(prhs[0]);
    mexPrintf("y-dim=%d, n = %d, p=%d\n", mxGetM(prhs[1]), n, p);
    
    if (n != mxGetM(prhs[1])) {
        mexErrMsgTxt("y does not have the correct dimension!");
    }
    
    if (nthread>0) {
        #ifndef DISABLE_OMP
        omp_set_num_threads(nthread);
        mexPrintf("# OMP threads = %d\n", nthread);
        #endif
    }
    
    // Compute number of features per partition
    std::div_t dvresult = div(p, nthread);
    int p1 = dvresult.quot; // first nthread-1 partitions
    int p2 = p1+dvresult.rem; // last partition
    
    // Partition X (column-major)
    // Find such lambda that if used for regularization,
    // optimum would have all weights zero.
    std::vector<data_partition> partitions;
    double lambda_max;
    int i=0;
    //#pragma omp parallel for
    for(int ip=0; ip<nthread; ip++) {
        int ncol = (ip<nthread-1)? p1:p2;
        data_partition part(n, ncol, nlambda);
        for(int col=0; col<ncol; col++) {
            double xty = 0, ssq = 0;
            sparse_array &vec = part.pcols[col];
            for(int row=0; row<n; row++) {
                double val = X[i++];
                if (val != 0) {
                    xty += val*y[row];
                    ssq += val*val;
                    vec.add(row, val);
                }
            }
            xty = std::abs(xty);
            part.colssq[col] = ssq;
            
            // normalization
            if (normalize && ssq) {
                double scale = sqrt(ssq);
                for (int idx=0; idx<vec.length(); idx++) {
                    vec.vals[idx] = vec.vals[idx]/scale;
                }
                xty = xty/scale;
            }
            
            if (xty > lambda_max) lambda_max = xty;
        }
        for(int row = 0; row<n; row++) {
            part.r[row] = y[row];
        }
        partitions.push_back(part);
        //partitions[ip] = part;
    }
    mexPrintf("Partitioned X\n");
    
    plhs[1] = mxCreateDoubleMatrix(1, nlambda+1, mxREAL);
    double *Lambda = mxGetPr(plhs[1]);
    Lambda[0] = lambda_max;
    mexPrintf("Max Lambda = %f\n", lambda_max);
    double multiplier = exp(log(lambda_min_ratio)/nlambda);
    for(int l=1; l<nlambda+1; l++) {
        Lambda[l] = Lambda[l-1]*multiplier;
    }
    /*
    for(int ip=0; ip<nthread; ip++) {
        mexPrintf("%f\n", partitions[ip].norm_wp);
    }

    int idx_start=0;
    for(int ip=0; ip<nthread; ip++) {
        int dim = partitions[ip].ncol;
        mexPrintf("Partition %d\n", ip);
        for(int col=0; col<dim; col++) {
            mexPrintf("Col %d:", idx_start+col);
            sparse_array &vec = partitions[ip].pcols[col];
            for(int i=0; i<vec.length(); i++) {
                mexPrintf(" (%d, %f)", vec.idxs[i], vec.vals[i]);
            }
            mexPrintf("\n");
        }
        idx_start += dim;
    }
    
    double change[nthread];
    #pragma omp parallel for
    for(int ip=0; ip<nthread; ip++) {
        data_partition *part = &partitions[ip];
        change[ip] = cd_local(part, lambda);
    }
    for(int ip=0; ip<nthread; ip++) {
        mexPrintf("Partition %d: change = %f, weight = %f\n", ip, change[ip], partitions[ip].wp[0]);
    }*/
    
    std::vector<data_partition> &partitions_ref = partitions;
    pcd_global(partitions_ref, y, method, gamma, Lambda, n, nlambda, normalize, threshold, maxiter, nthread);
    plhs[0] = mxCreateDoubleMatrix(p, nlambda, mxREAL);
    
    double *w = mxGetPr(plhs[0]);
    i=0;
    // when normalization is applied before computation, 
    // weights are postprocessed to fit the original data
    if (normalize) {
        for(int l=0; l<nlambda; l++) {
            for(int ip=0; ip<nthread; ip++) {
                data_partition &part = partitions[ip];
                int idx_start = l*part.ncol;
                for(int j=0; j<part.ncol; j++) {
                    w[i++] = part.wp[idx_start+j]/sqrt(part.colssq[j]);
                }
            }
        }
    } else {
        for(int l=0; l<nlambda; l++) {
            for(int ip=0; ip<nthread; ip++) {
                data_partition &part = partitions[ip];
                int idx_start = l*part.ncol;
                for(int j=0; j<part.ncol; j++) {
                    w[i++] = part.wp[idx_start+j];
                }
            }
        }
    }
    mexPrintf("Well done!\n");
}