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
#ifndef FORMULA_H_
#define FORMULA_H_

#include "linear.hpp"

///
///
class ProblemBase
{
public:
    virtual ~ProblemBase(void) {}

    virtual double loss() = 0;
    virtual void gradient() = 0;

};

///
///
class L2R_LR_Problem : public ProblemBase
{
private:
    const DatasetPtr dataset_;
    /** buffer for wTX result */
    RowVector wTX;
public:
    L2R_LR_Solver(const DatasetPtr);
    ~L2R_LR_Solver();

    double loss(const ColVector, const vector<double>);
    RowVector gradient(const ColVector, const vector<double>);
};
#endif// FORMULA_H_
