// Quardrtic and Linear Problem Optimizer
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#ifndef SOLVER_H_
#define SOLVER_H_
#include "linear.hpp"
#include "formula.hpp"

/// Base class for solvers
///
///
class SolverBase
{
public:
    virtual ~SolverBase();
    virtual void solve(ProblemPtr) = 0;
};

/// Gradint descent optimizer
///
class GradientDescent: public SolverBase
{
public:
    ~GradientDescent();
    void solve(ProblemPtr);
};

/// Stochastic gradient descent optimizer
class StochasticGD: public SolverBase
{
public:
    ~StochasticGD();
    void solve(ProblemPtr);
};

/// Limited BFGS optimizer
class L_BFGS: public SolverBase
{
public:
    ~L_BFGS();
    void solve(ProblemPtr);
};

/// Newton conjugate gradient optimizer
class NewtonCG: public SolverBase
{
public:
    ~NewtonCG();
    void solve(ProblemPtr);
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
    void solve(ProblemPtr);
};
#endif// SOLVER_H_
