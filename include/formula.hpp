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
    explicit Problem(DatasetPtr, const std::vector<double>&);
    virtual ~Problem(void) {};

    virtual double loss(const Eigen::Ref<const ColVector>&) = 0;
    virtual ColVector gradient(const Eigen::Ref<const ColVector>&) = 0;

protected:
    std::vector<double> C_;
    const DatasetPtr dataset_;
};

// Define the problem smart pointer type for polymorphism
typedef std::shared_ptr<Problem> ProblemPtr;

/// L1-Regularized Loss Logistic Regression
///
class L1R_LR_Problem : public Problem
{

private:
    /** g are some reusable part of the processes*/
    ColVector g_;

public:
    explicit L1R_LR_Problem(DatasetPtr, const std::vector<double>&);
    ~L1R_LR_Problem();

    double loss(const Eigen::Ref<const ColVector>&);
    ColVector gradient(const Eigen::Ref<const ColVector>&);
};
/// L2-Regularized Loss Logistic Regression
///
class L2R_LR_Problem : public Problem
{

private:
    /** g are some reusable part of the processes*/
    ColVector g_;

public:
    explicit L2R_LR_Problem(DatasetPtr, const std::vector<double>&);
    ~L2R_LR_Problem();

    double loss(const Eigen::Ref<const ColVector>&);
    ColVector gradient(const Eigen::Ref<const ColVector>&);
};

} // oplin

#endif// OPENLINEAR_FORMULA_H_
