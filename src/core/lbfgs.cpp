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
LBFGS::LBFGS(const size_t m_step=10): m_step_(m_step) {}
LBFGS::~LBFGS(){}

void
LBFGS::two_loop(ProblemPtr problem, Eigen::Ref<ColVector>& w)
{

}
void
LBFGS::search_direction(ProblemPtr problem, ParamPtr param, Eigen::Ref<ColVector>& w)
{

}
void
LBFGS::update(ProblemPtr problem, ParamPtr param, Eigen::Ref<ColVector>& w)
{

}

void
LBFGS::solve(ProblemPtr problem, ParamPtr param, Eigen::Ref<ColVector>& w)
{

    // initializations
    loss_ = problem->loss(w);
    grad_ = ColVector::Zero(problem->dataset_->dimension,1);
    prev_w_ = w;
    prev_loss_ = loss_;
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
        /// 01 - Update gradient and line search direction p
        problem->gradient(prev_w_, grad_);
        // two loop to get search direction
        search_dirction(problem, param, w);
        alpha = 1.0;
        //     |- line search on p and update w and loss values
        iter = this->line_search(problem, w, alpha);


        /// 02 - Termination Check
        rela_improve = fabs(loss_ - prev_loss_);
        VOUT("|%5d|%15.4f|%15.6f|%5d|\n",epoch_,loss_,rela_improve,iter);
        if(rela_improve < param->rela_tol || loss_ < param->abs_tol) break;

        update(problem, param, w);
        /// 03 - Update varaiables
        prev_loss_ = loss_;
        w.swap(prev_w_);
    }
}

} // oplin
