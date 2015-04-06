// Quardrtic Problem Optimizer
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#ifndef SOLVER_H_
#define SOLVER_H_
#include "linear.hpp"

/// Base class for solvers
///
///
class SolverBase
{
public:
    virtual ~SolverBase();
    virtual void solve(ProblemBase) = 0;
};

/// Gradint descent solver
///
class GradientDescent: public SolverBase
{
public:
    ~GradientDescent();
    void solve(ProblemBase);
};

/// Stochastic gradient descent Solver
class StochasticGD: public SolverBase
{
    ~StochasticGD();
    void solve(ProblemBase);
};

/// Limited BFGS Solver
class L_BFGS: public SolverBase
{
public:
    ~L_BFGS();
    void solve(ProblemBase);
};

/// NewtonCG Optimizer
class NewtonCG: public SolverBase
{
    ~NewtonCG();
    void solve(ProblemBase);
};


#endif// SOLVER_H_
