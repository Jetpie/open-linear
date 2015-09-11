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
namespace oplin{
using std::cout;
using std::endl;
using std::cerr;
using namespace Eigen;
Problem::Problem(DatasetPtr dataset, const std::vector<double>& C) : dataset_(dataset)
{
    // use swap trick
    std::vector<double>(C).swap(C_);
    // C_ = C;
}
void
Problem::update_weights(Eigen::Ref<ColVector> new_w, const Eigen::Ref<const ColVector>& w,
                                const Eigen::Ref<const ColVector>& p, const double& alpha)
{
    new_w.noalias() = w + alpha * p;
}


/*********************************************************************
 *                                  L1-Regularized Logistic Regression
 *********************************************************************/
L1R_LR_Problem::L1R_LR_Problem(DatasetPtr dataset, const std::vector<double>& C) : Problem(dataset, C)
{
    z_ = ColVector(dataset->n_samples, 1);
}

L1R_LR_Problem::~L1R_LR_Problem(){}

/**
 * Compute the L1-regularized loss functionn
 *
 * @param w weights
 *
 */
double
L1R_LR_Problem::loss(const Eigen::Ref<const ColVector>& w)
{
    // loss value, initilization
    // l1-norm regularization term
    double f = w.lpNorm<1>();

    std::vector<double> y = dataset_->y;

    // W^T X
    z_.noalias() = w.transpose() * (*(dataset_->X));

    // loss function : negative log likelihood
    for(size_t i = 0; i < dataset_->n_samples; ++i)
        f += C_[i] * log( 1 + exp(-y[i] * z_(i) ) );


    return f;
}

/**
 * Compute the L1-regularized gradient descent direction
 *
 * Reference:
 * Galen Andrew and Jianfeng Gao. 2007. Scalable training of L1-regularized \
 * log-linear models. In ICML.
 *
 * @param w weights
 */
void
L1R_LR_Problem::gradient(const Eigen::Ref<const ColVector>& w, Eigen::Ref<ColVector> grad)
{
    const std::vector<double> y = dataset_->y;

    for(size_t i = 0; i < dataset_->n_samples; ++i)
    {
        // h_w(y_i,x_i) - sigmoid function
        z_(i) = 1 / (1+exp(-y[i]*z_(i)));
        // C * (h_w(y_i,x_i) - 1) * y[i]
        z_(i) = C_[i]*(z_(i)-1)*y[i];
    }
    // declare noalias here to enforce lazy evaluation for memory saving
    grad.noalias() = *(dataset_->X) * z_;


    // this is the key part of the well defined pesudo-gradient in
    // Jianfeng et al's paper
    for(size_t i=0; i < dataset_->dimension;++i)
    {
        if(w(i) == 0)
        {
            if(grad(i) < -1 ) ++grad(i);
            else if(grad(i) > 1) --grad(i);
            else grad(i) = 0;
        }
        else
        {
            grad(i) += w(i) > 0? 1 : (-1);
        }
    }

}
void
L1R_LR_Problem::update_weights(Eigen::Ref<ColVector> new_w, const Eigen::Ref<const ColVector>& w,
                               const Eigen::Ref<const ColVector>& p, const double& alpha)
{
    new_w.noalias() = w + alpha * p;
    // prevent moving outside orthants
    for(size_t i = 0; i < dataset_->dimension; ++i)
    {
        // check same sign
        if(new_w(i) * w(i) < 0 ) new_w(i) = 0.0;
    }
}
/*********************************************************************
 *                                  L2-Regularized Logistic Regression
 *********************************************************************/
L2R_LR_Problem::L2R_LR_Problem(DatasetPtr dataset, const std::vector<double>& C) : Problem(dataset, C)
{
    z_ = ColVector(dataset->n_samples, 1);
}

L2R_LR_Problem::~L2R_LR_Problem(){}

/**
 * Compute the L2-regularized loss functionn
 *
 * @param w
 * @param C
 *
 */
double
L2R_LR_Problem::loss(const Eigen::Ref<const ColVector>& w)
{
    // loss value, initilization
    // l2-norm regularization term
    double f = w.squaredNorm() / 2.0;

    std::vector<double> y = dataset_->y;

    // W^T X
    z_.noalias() = w.transpose() * (*(dataset_->X));

    // loss function : negative log likelihood
    for(size_t i = 0; i < dataset_->n_samples; ++i)
        f += C_[i] * log( 1 + exp(-y[i] * z_(i) ) );


    return f;
}

/**
 * Compute the L2-regularized gradient descent direction
 *
 *
 */
void
L2R_LR_Problem::gradient(const Eigen::Ref<const ColVector>& w, Eigen::Ref<ColVector> grad)
{
    const std::vector<double> y = dataset_->y;

    for(size_t i = 0; i < dataset_->n_samples; ++i)
    {
        // h_w(y_i,x_i) - sigmoid function
        z_(i) = 1 / (1+exp(-y[i]*z_(i)));
        // C * (h_w(y_i,x_i) - 1) * y[i]
        z_(i) = C_[i]*(z_(i)-1)*y[i];

    }
    // declare noalias here to enforce lazy evaluation for memory saving
    grad.noalias() = (*(dataset_->X)) * z_ ;
    grad = grad + w;

}

} // oplin
