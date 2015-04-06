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

/// Gradint descent solver
///
class GradientDescent: public SolverBase
{
public:
    ~GradientDescent();
    void solve(ProblemPtr);
};

/// Stochastic gradient descent Solver
class StochasticGD: public SolverBase
{
    ~StochasticGD();
    void solve(ProblemPtr);
};

/// Limited BFGS Solver
class L_BFGS: public SolverBase
{
public:
    ~L_BFGS();
    void solve(ProblemPtr);
};

/// NewtonCG Optimizer
class NewtonCG: public SolverBase
{
    ~NewtonCG();
    void solve(ProblemPtr);
};


#endif// SOLVER_H_
