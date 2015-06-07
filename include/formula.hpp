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
#ifndef OPENLINEAR_FORMULA_H_
#define OPENLINEAR_FORMULA_H_

#include "linear.hpp"

namespace oplin{

///
///
class Problem
{

public:
    virtual ~Problem(void) {};

    virtual double loss(const Eigen::Ref<const ColVector>&, const std::vector<double>&) = 0;
    virtual ColVector gradient(const Eigen::Ref<const ColVector>&, const std::vector<double>&) = 0;

};

// Define the problem smart pointer type for polymorphism
typedef std::shared_ptr<Problem> ProblemPtr;

/// L1-Regularized Loss Logistic Regression
///
class L1R_LR_Problem : public Problem
{

private:
    const DatasetPtr dataset_;
    /** g are some reusable part of the processes*/
    ColVector g_;

public:
    explicit L1R_LR_Problem(const DatasetPtr);
    ~L1R_LR_Problem();

    double loss(const Eigen::Ref<const ColVector>&,const std::vector<double>&);
    ColVector gradient(const Eigen::Ref<const ColVector>&,const std::vector<double>&);
};
/// L2-Regularized Loss Logistic Regression
///
class L2R_LR_Problem : public Problem
{

private:
    const DatasetPtr dataset_;
    /** g are some reusable part of the processes*/
    ColVector g_;

public:
    explicit L2R_LR_Problem(const DatasetPtr);
    ~L2R_LR_Problem();

    double loss(const Eigen::Ref<const ColVector>&,const std::vector<double>&);
    ColVector gradient(const Eigen::Ref<const ColVector>&,const std::vector<double>&);
};

} // oplin

#endif// OPENLINEAR_FORMULA_H_
