#include "common.h"

double soft_threshold(double z, double lambda) {
    if (z > lambda) return z - lambda;
    if (z < -lambda) return z + lambda;
    else return 0;
}

double lasso(double z, double w, double ssq, double lambda, int normalize) {
    if (normalize) {
        z = z+w;
        return soft_threshold(z, lambda);
    } else {
        z = z+ssq*w;
        return soft_threshold(z, lambda)/ssq;
    }
}

double MCP(double z, double w, double gamma, double lambda) {
    z = z+w;
    double z_abs = abs(z);
    if (z_abs <= lambda) return 0;

    double s = 0;
    if (z > 0) s = 1;
    else if (z < 0) s = -1;
    if (z_abs <= gamma*lambda) return s*(z_abs-lambda)/(1-1/gamma);
    else return z;
}

double SCAD(double z, double w, double gamma, double lambda) {
    z = z+w;
    double z_abs = abs(z);
    if (z_abs <= lambda) return 0;

    double s = 0;
    if (z > 0) s = 1;
    else if (z < 0) s = -1;
    if (z_abs <= 2*lambda) return s*(z_abs-lambda);
    else if (z_abs <= gamma*lambda) return s*(z_abs-gamma*lambda/(gamma-1))/(1-1/(gamma-1));
    else return z;
}

// Run coordinate descent on a data partition,
// and return a measurement of change in weights
double cd_local(data_partition &part, int method, double gamma, double lambda, int normalize, int idx_start) {
    double change = 0;
    for(int j=0; j<part.ncol; j++) {
        sparse_array &vec = part.pcols[j];
        double z = 0;
        double w_old = part.wp[idx_start+j];
        for(int i=0; i<vec.length(); i++) {
            z += vec.vals[i]*part.r[vec.idxs[i]];
        }
        double w_new;
        // MCP and SCAD are implemented only for normalized data
        if (method == 1) {
            w_new = MCP(z, w_old, gamma, lambda);
        } else if (method == 2) {
            w_new = SCAD(z, w_old, gamma, lambda);
        } else if (method == 3) {
            w_new = lasso(z, w_old, part.colssq[j], lambda, normalize);
        }
        //z += part.colssq[j]*w_old;
        //double w_new = soft_threshold(z, lambda)/part.colssq[j];
        double diff = w_new - w_old;
        //mexPrintf("z = %f, w_new = %f, w_old = %f, colssq = %f\n", z, w_new, w_old, part.colssq[j]);
        if (diff != 0) {
            for(int i=0; i<vec.length(); i++) {
                int idx = vec.idxs[i];
                double v = vec.vals[i]*diff;
                part.Xwp[idx] += v;
                part.r[idx] -= v;
            }
            part.wp[idx_start+j] = w_new;
            //mexPrintf("%f\n", part.wp[j]);
        }
        double descent = diff*diff;
        if (!normalize) descent *= part.colssq[j];
        if (descent > change) change = descent;
    }
    return change;
}

// parallel coordinate descent
void pcd_global(std::vector<data_partition> &partitions, double *y, int method, double gamma, double *Lambda, int n, int nlambda, int normalize, double threshold, int maxiter, int nthread) {
    double null_obj = 0;
    for(int i=0; i<n; i++) null_obj += y[i]*y[i]; // unscaled initial objective value
    for(int l=0; l<nlambda; l++) {
        // warm-start
        if(l>0) {
            for(int ip=0; ip<nthread; ip++) {
                data_partition &part = partitions[ip];
                int ncol = part.ncol;
                int idx_current = l*ncol, idx_prev = (l-1)*ncol;
                for(int j=0; j<ncol; j++) {
                    part.wp[idx_current+j] = part.wp[idx_prev+j];
                }
            }
        }
        // optimization
        // Weights are all zero for Lambda[0],
        // so computation starts at Lambda[1]
        double lambda = Lambda[l+1];
        mexPrintf("lambda = %f\n", lambda);
        for(int i=0; i<maxiter; i++) {
            double maxChange = 0;
            #pragma omp parallel for reduction(max:maxChange)
            for(int ip=0; ip<nthread; ip++) {
                data_partition &part = partitions[ip];
                int idx_start = l*part.ncol;
                double change = cd_local(part, method, gamma, lambda, normalize, idx_start);
                if (change > maxChange) {
                    maxChange = change;
                }
            }
            maxChange = maxChange/null_obj;
            mexPrintf("maxChange = %f\n", maxChange);
            
            // Update residuals for the global problem
            double r[n];
            for(int row=0; row<n; row++) r[row] = y[row];
            for(int ip=0; ip<nthread; ip++) {
                data_partition &part = partitions[ip];
                for(int row=0; row<n; row++) {
                    r[row] -= part.Xwp[row];
                }
            }
            
            // Broadcast residuals to all partitions
            for(int ip=0; ip<nthread; ip++) {
                data_partition &part = partitions[ip];
                for(int row=0; row<n; row++) {
                    part.r[row] = r[row];
                }
            }
            // Check convergence
            if (maxChange < threshold) {
                mexPrintf("Lambda %d: converged\n", l+1);
                break;
            }
        }
    }
}





































