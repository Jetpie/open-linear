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

Problem::Problem(DatasetPtr dataset, const std::vector<double>& C) : dataset_(dataset)
{
    // use swap trick
    std::vector<double>(C).swap(C_);
    regularizer_ = NULL;
}

void
Problem::regularized_gradient(const Eigen::Ref<const ColVector>& w, Eigen::Ref<ColVector> grad)
{
    if(regularizer_) regularizer_->gradient(w,grad);
}

void
Problem::update_weights(Eigen::Ref<ColVector> new_w, const Eigen::Ref<const ColVector>& w,
                                const Eigen::Ref<const ColVector>& p, const double& alpha)
{
    new_w.noalias() = w + alpha * p;
}

double
L1_Regularizer::loss(const Eigen::Ref<const ColVector>& w)
{

    return w.lpNorm<1>();
}

/**
 * Compute the L1-regularized gradient
 *
 * Reference:
 * Galen Andrew and Jianfeng Gao. 2007. Scalable training of L1-regularized \
 * log-linear models. In ICML.
 *
 * @param w weights
 */
void
L1_Regularizer::gradient(const Eigen::Ref<const ColVector>& w, Eigen::Ref<ColVector> grad)
{
    // this is the key part of the well defined pesudo-gradient in
    // Jianfeng et al's paper
    for(int i=0; i < grad.rows(); ++i)
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

double
L2_Regularizer::loss(const Eigen::Ref<const ColVector>& w)
{
    return (w.squaredNorm() * 0.5);
}

void
L2_Regularizer::gradient(const Eigen::Ref<const ColVector>& w, Eigen::Ref<ColVector> grad)
{
    grad.noalias() += w;
}

/*********************************************************************
 *                                                 Logistic Regression
 *********************************************************************/
LR_Problem::LR_Problem(DatasetPtr dataset, const std::vector<double>& C) : Problem(dataset, C)
{
    z_ = ColVector(dataset->n_samples, 1);
}
LR_Problem::~LR_Problem(){}

/**
 * Compute the loss functionn
 *
 * @param w weights
 *
 */
double
LR_Problem::loss(const Eigen::Ref<const ColVector>& w)
{

    double f = regularizer_? regularizer_->loss(w):0;

    const std::vector<double>& y = dataset_->y;

    // W^T X
    z_.noalias() = w.transpose() * (*(dataset_->X));
    // std::cout << z_.sum() << std::endl;
    // loss function : negative log likelihood
    for(size_t i = 0; i < dataset_->n_samples; ++i)
        f += C_[i] * log( 1 + exp(-y[i] * z_(i) ) );

    return f;
}


/**
 * Compute the gradient descent direction
 *
 * @param w weights
 */
void
LR_Problem::gradient(const Eigen::Ref<const ColVector>& w, Eigen::Ref<ColVector> grad)
{
    const std::vector<double>& y = dataset_->y;

    for(size_t i = 0; i < dataset_->n_samples; ++i)
    {
        // h_w(y_i,x_i) - sigmoid function
        z_(i) = 1 / (1+exp(-y[i]*z_(i)));
        // C * (h_w(y_i,x_i) - 1) * y[i]
        z_(i) = C_[i]*(z_(i)-1)*y[i];
    }

    // declare noalias here to enforce lazy evaluation for memory saving
    grad.noalias() = *(dataset_->X) * z_;
}

/*********************************************************************
 *                                  L1-Regularized Logistic Regression
 *********************************************************************/
L1R_LR_Problem::L1R_LR_Problem(DatasetPtr dataset, const std::vector<double>& C) : LR_Problem(dataset, C)
{
    regularizer_ = std::make_shared<L1_Regularizer>();
    if(!regularizer_)
    {
        cerr << "L1R_LR_Problem::L1R_LR_Problem : Failed to declare regularizer! ("
             << __FILE__ << ", line " << __LINE__ << ")."<< endl;
        throw(std::bad_alloc());
    }
}
L1R_LR_Problem::~L1R_LR_Problem(){}

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
L2R_LR_Problem::L2R_LR_Problem(DatasetPtr dataset, const std::vector<double>& C) : LR_Problem(dataset, C)
{
    regularizer_ = std::make_shared<L2_Regularizer>();
    if(!regularizer_)
    {
        cerr << "L2R_LR_Problem::L2R_LR_Problem : Failed to declare regularizer! ("
             << __FILE__ << ", line " << __LINE__ << ")."<< endl;
        throw(std::bad_alloc());
    }
}
L2R_LR_Problem::~L2R_LR_Problem(){}

} // oplin
