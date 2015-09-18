// L-BFGS
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#include "solver.hpp"


namespace oplin
{
LBFGS::LBFGS() : m_step_(10) {}
LBFGS::LBFGS(const size_t m_step): m_step_(m_step) {}
LBFGS::~LBFGS(){}

/**
 * Compute the approximate Hessian using two loop recursion
 *
 * @param problem
 * @param w
 *
 */
void
LBFGS::two_loop(ProblemPtr problem, const Eigen::Ref<const ColVector>& w)
{
    const size_t num = s_list_.size();
    if(!num) return;

    std::vector<double> alpha(num);
    size_t i;
    for(i= num - 1; i >0 ;--i)
    {
        alpha[i] = ro_list_[i] * (double)((*s_list_[i]).transpose() * p_);
        p_.noalias() = p_ - alpha[i] * (*y_list_[i]);
    }

    const double gamma = ro_list_[num-1] / y_list_[num-1]->squaredNorm();
    p_.noalias() = gamma *  p_;
    double beta;

    for(i = 0; i < num; ++i)
    {
        beta = ro_list_[i] * (double)((*y_list_[i]).transpose() * p_);
        p_.noalias() = p_ + * s_list_[i] * (alpha[i] - beta);
    }
}

/**
 * Compute search direction by multiply inverse Hessian and steepest gradient
 *
 * @param problem
 * @param param
 * @param w
 */
void
LBFGS::search_direction(ProblemPtr problem, ParamPtr param, const Eigen::Ref<const ColVector>& w)
{
    p_ = steepest_grad_;
    two_loop(problem, w);

    // limit the search direction for l1 norm
    if(param->problem_type == 0)
    {
        for(int i=0; i < w.rows(); ++i)
        {
            if(steepest_grad_[i] * p_[i] <= 0)
            {
                p_[i] = 0;
            }
        }
    }

    p_.noalias() = (-1) * p_;
}

/**
 * Update the ro, y and s list for two loop
 *
 * @param problem
 * @param param
 * @param w
 */
void
LBFGS::update(ProblemPtr problem, ParamPtr param, const Eigen::Ref<const ColVector>& w)
{
    const size_t num = s_list_.size();
    ColVectorPtr s_k = NULL, y_k = NULL;

    // check if more than m steps of vectors have been stored
    if(num < m_step_)
    {
        // declare new vector
        s_k = std::make_shared<ColVector>(w.rows());
        y_k = std::make_shared<ColVector>(w.rows());

    }
    // this includes the sanity check for s_k and y_k
    if(!s_k || !y_k)
    {
        // the oldest ptr in the queue
        s_k = s_list_.front();
        y_k = y_list_.front();

        // pop old steps
        y_list_.pop_front();
        s_list_.pop_front();
        ro_list_.pop_front();
    }

    (*s_k).noalias() = next_w_ - w;
    (*y_k).noalias() = next_grad_ - grad_;
    double ro_k = (*y_k).transpose() * (*s_k);

    if(ro_k == 0)
    {
        std::cout << "Warning: ro_k is unexpectedly divide by zero(y_k^T * s_k)."
            " Auto fixed (compile with debug to see detail)!" << std::endl;
        VOUT("************************************************************\n"
             "ro_k = 1 / (s_k.traspose() * y_k): The reason cause dividend\n"
             "to be zero is mostly gradients between two epochs is too close to zero.\n"
             "Fix this by set ColVector::Ones(y_k->rows())to y_k.\n"
             "************************************************************\n");
        *y_k = ColVector::Ones(y_k->rows());
        ro_k = (*y_k).transpose() * (*s_k);
    }

    ro_k = 1 / ro_k;

    y_list_.push_back(y_k);
    s_list_.push_back(s_k);
    ro_list_.push_back(ro_k);


}

void
LBFGS::solve(ProblemPtr problem, ParamPtr param, Eigen::Ref<ColVector>& w)
{

    // initializations
    loss_ = problem->loss(w);

    grad_ = ColVector::Zero(w.rows(),1);
    problem->gradient(w, grad_);
    if(param->problem_type == 0)
    {
        steepest_grad_ = grad_;
        problem->regularized_gradient(w, steepest_grad_);
    }
    else
    {
        problem->regularized_gradient(w,grad_);
        steepest_grad_ = grad_;
    }

    next_grad_ = ColVector::Zero(w.rows(),1);
    next_loss_ = loss_;
    next_w_ = w;

    double rela_improve = 0;
    double alpha;
    size_t iter = 0;

    // check if the weights already optimized
    if(loss_ < param->abs_tol)
    {
        return;
        VOUT("Already optimized weights get!");
    }

    // -- debug print
    VOUT("\n*** To disable debug info: $ make DISABLE_DEBUG=yes ***\n");
    VOUT("|%5s|%15s|%15s|%5s|\n","Epoch","Loss","Improve","#iter");

    for(epoch_ = 0; epoch_ < param->max_epoch; ++epoch_)
    {

        /*** Iteration - k ***/
        // w = w_k, grad = grad_k
        // next_w = w_{k+1}, next_grad = grad_{k+1}

        /// 01 - Update line search direction p
        //     |- the typical form of p will be p_k = -B_k^(-1) * grad(f_k)
        search_direction(problem, param, w);
        //     |- line search on p and update w and loss values
        alpha = 1.0;
        iter = this->line_search(problem, w, alpha);


        /// 02 - Termination Check
        rela_improve = fabs((next_loss_ - loss_) / loss_);
        VOUT("|%5d|%15.4f|%15.6f|%5d|\n",epoch_,next_loss_,rela_improve,iter);
        if(rela_improve < param->rela_tol || next_loss_ < param->abs_tol)
        {
            // assign next_w_ to w as return value
            w.swap(next_w_);
            break;
        }

        /// 03 - Update varaiables
        problem->gradient(next_w_, next_grad_);

        if(param->problem_type == 0)
        {
            steepest_grad_ = next_grad_;
            problem->regularized_gradient(next_w_, steepest_grad_);
        }
        else
        {
            problem->regularized_gradient(next_w_,next_grad_);
            steepest_grad_ = next_grad_;
        }
        // problem->regularized_gradient(next_w_,next_grad_);
        // steepest_grad_ = next_grad_;

        update(problem, param, w);

        loss_ = next_loss_;
        w.swap(next_w_);
        grad_.swap(next_grad_);
    }
}

} // oplin
