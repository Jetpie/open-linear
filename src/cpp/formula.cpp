// Problem formulations
//
// Naming Convention:
// 1.solver classes are named by all capital letter denote type and a
//   "Solver" prefix separated by underscore.
// 2.some math notation like w^t by x will be denoted as wTx,
//   which big T means transpose. For the same style, other name
//   convention can be deduced like wT means transpose of w. It worths
//   something to mention that big X means matrix and small x means
//   vector. Hope this is helpful for readers.
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#include "formula.hpp"

/**
 * Compute the loss function of logistic regression
 *
 * @param w
 * @param C
 *
 */
double
L2R_LR_Problem::loss(const ColVector& w, const vector<double>& C)
{
    // loss value, initilization
    // l2-norm regularization term
    double f = w.squaredNorm() / 2.0;

    size_t n_samples = dataset_->n_samples;
    vector<double> y = dataset_->y;

    // W^T X
    g_ = w.transpose() * (*(dataset_->X));

    for(size_t i = 0; i < n_samples; ++i)
    {
        double ywTX = y[i] * g_(i);
        // loss function : negative log likelihood
        f += C[i] * log( 1 + exp(-ywTX) );
    }
    return f;
}

/**
 * Compute the gradient
 *
 *
 */
RowVector
L2R_LR_Problem::gradient(const ColVector& w, const vector<double>& C)
{
    size_t n_samples = dataset_->n_samples;
    vector<double> y = dataset_->y;

    for(size_t i = 0; i < n_samples; ++i)
    {
        // h_w(y_i,x_i) - sigmoid function
        g_(i) = 1 / ( 1 + exp( -y[i] * g_(i) ) );
        // C * (h_w(y_i,x_i) - 1) * y[i]
        g_(i) = C[i] * (g_(i) - 1) * y[i];
    }
    ColVector grad = (*(dataset_->X)) * g_;
    grad += w;

    return grad;
}
