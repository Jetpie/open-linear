// Problem Optimizer in iterative ways
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#ifndef OPENLINEAR_SOLVER_H_
#define OPENLINEAR_SOLVER_H_
#include <deque>
#include "linear.hpp"
#include "formula.hpp"
namespace oplin{

enum LineSearchCondition
{
    CONSTANT,
    ARMIJO_CONDITION,
    WHOLF_CONDITION
};
/// Base class for solvers
///
///
class SolverBase
{
public:
    virtual ~SolverBase();
    virtual void solve(ProblemPtr, ParamPtr, Eigen::Ref<ColVector>&) = 0;

protected:
    /** buffer for loss value at iteration k */
    double loss_;
    /** buffer for loss value at iteration k+1 */
    double next_loss_;
    /** column vector to store steepest gradient k */
    ColVector steepest_grad_;
    /** column vector to store steepest gradient at iteration k */
    ColVector grad_;
    /** column vector to store steepest gradient at iteration k+1 */
    ColVector next_grad_;
    /** column vector to store steepest gradient at iteration k+1 */
    ColVector next_w_;
    /** column vector to store search direction*/
    ColVector p_;
    /** epoch */
    size_t epoch_;

    virtual size_t line_search(const ProblemPtr, Eigen::Ref<ColVector>, double& alpha);
};

/// Gradint descent optimizer
///
class GradientDescent: public SolverBase
{
public:
    ~GradientDescent();
    void solve(ProblemPtr, ParamPtr, Eigen::Ref<ColVector>&);
};

/// Stochastic gradient descent optimizer
class StochasticGD: public SolverBase
{
public:
    ~StochasticGD();
};

class LBFGS: public SolverBase
{
public:
    LBFGS();
    LBFGS(const size_t);
    ~LBFGS();
    void solve(ProblemPtr, ParamPtr, Eigen::Ref<ColVector>&);

private:

    void two_loop(ProblemPtr, const Eigen::Ref<const ColVector>&);
    void search_direction(ProblemPtr, const Eigen::Ref<const ColVector>&);
    void update(ProblemPtr, ParamPtr, const Eigen::Ref<const ColVector>&);


    /** m steps to keep */
    size_t m_step_;

    std::deque<ColVectorPtr> y_list_;
    std::deque<ColVectorPtr> s_list_;
    std::deque<double> ro_list_;

};

/// Newton trust region optimizer
///
/// Reference:
/// Chih-Jen Lin, Ruby C. Weng, and S. Sathiya Keerthi.
/// Trust region Newton method for large-scale logistic regression.
/// Journal of Machine Learning Research, 9:627–650, 2008.
///
class TRON: public SolverBase
{
public:
    ~TRON();
};

} // oplin

#endif// OPENLINEAR_SOLVER_H_
