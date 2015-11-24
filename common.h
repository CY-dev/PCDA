#include <cmath>
#include <iostream>
#include <memory.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <queue>
#include <list>
#include <map>

#include "mex.h"

#ifndef DISABLE_OMP
#include <omp.h>
#endif

//sparse array
struct sparse_array {
    std::vector<unsigned int> idxs;
    std::vector<double> vals;
    
    int length() {
        return idxs.size();
    }
    void add(unsigned int idx, double val) {
        idxs.push_back(idx);
        vals.push_back(val);
    }
};

struct data_partition {
    // sparse column vectors for the partition (Xp),
    // normalized when "normalize = 1"
    std::vector<sparse_array> pcols;

    int nrow; // Number of data points
    int ncol; // Number of features for the partition
    int nlambda; // Number of values for regularization parameter lambda
    
    // residuals
    double *r;
    
    double *wp; // Matrix of weights for the partition (ncol by nlambda)
    double *Xwp; // Xp*wp
    
    // column sum of squares for the unnormalized data
    double *colssq;
    
    data_partition(){
        nrow = 0;
        ncol = 0;
        nlambda = 0;
        r = new double[0];
        wp = new double[0];
        Xwp = new double[0];
        colssq = new double[0];
    };
    
    data_partition(int _nrow, int _ncol, int _nlambda) {
        nrow = _nrow;
        ncol = _ncol;
        nlambda = _nlambda;
        pcols.resize(ncol);
        r = new double[nrow];
        wp = new double[ncol*nlambda]();
        Xwp = new double[nrow]();
        colssq = new double[ncol];
    }
    
    data_partition(const data_partition &that) {
        pcols = that.pcols;
        nrow = that.nrow;
        ncol = that.ncol;
        nlambda = that.nlambda;
        
        r = new double[nrow];
        wp = new double[ncol*nlambda];
        Xwp = new double[nrow];
        colssq = new double[ncol];
        
        std::copy(that.r, that.r+nrow, r);
        std::copy(that.wp, that.wp+ncol*nlambda, wp);
        std::copy(that.Xwp, that.Xwp+nrow, Xwp);
        std::copy(that.colssq, that.colssq+ncol, colssq);
    }
    
    data_partition& operator=(data_partition that) {
        swap(*this, that);
        return *this;
    }
    
    friend void swap(data_partition &p1, data_partition &p2) {
        using std::swap; // enable ADL
        
        swap(p1.pcols, p2.pcols);
        swap(p1.nrow, p2.nrow);
        swap(p1.ncol, p2.ncol);
        swap(p1.nlambda, p2.nlambda);
        swap(p1.r, p2.r);
        swap(p1.wp, p2.wp);
        swap(p1.Xwp, p2.Xwp);
        swap(p1.colssq, p2.colssq);
    }
    
    ~data_partition() {
        delete[] r;
        delete[] wp;
        delete[] Xwp;
        delete[] colssq;
    }
};

void pcd_global(std::vector<data_partition> &partitions, double *y, int method, double gamma, double *Lambda, int n, int nlambda, int normalize, double threshold, int maxiter, int nthread);