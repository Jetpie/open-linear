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
class Regularizer
{
public:
    virtual ~Regularizer(void) {};
    virtual double loss(const Eigen::Ref<const ColVector>&) = 0;
    virtual void gradient(const Eigen::Ref<const ColVector>&, Eigen::Ref<ColVector>) = 0;
};
typedef std::shared_ptr<Regularizer> RegularizerPtr;

///
class L1_Regularizer : public Regularizer
{
public:
    virtual ~L1_Regularizer(void) {};
    double loss(const Eigen::Ref<const ColVector>&);
    void gradient(const Eigen::Ref<const ColVector>&, Eigen::Ref<ColVector>);
};

class L2_Regularizer : public Regularizer
{
public:
    virtual ~L2_Regularizer(void) {};
    double loss(const Eigen::Ref<const ColVector>&);
    void gradient(const Eigen::Ref<const ColVector>&, Eigen::Ref<ColVector>);
};

/// A general convex regularization problem
///
class Problem
{
public:
    explicit Problem(DatasetPtr, const std::vector<double>&);
    virtual ~Problem(void) {};

    virtual double loss(const Eigen::Ref<const ColVector>&) = 0;
    virtual void gradient_Fx(const Eigen::Ref<const ColVector>&, Eigen::Ref<ColVector>) = 0;
    virtual void gradient_Rx(const Eigen::Ref<const ColVector>&, Eigen::Ref<ColVector>);
    virtual void update_weights(Eigen::Ref<ColVector>, const Eigen::Ref<const ColVector>&,
                                const Eigen::Ref<const ColVector>&, const double&);

    const DatasetPtr dataset_;

protected:
    std::vector<double> C_;
    RegularizerPtr regularizer_;
};



// Define the problem smart pointer type for polymorphism
typedef std::shared_ptr<Problem> ProblemPtr;






/// basic logistic regression
///
class LR_Problem : public Problem
{

public:
    explicit LR_Problem(DatasetPtr, const std::vector<double>&);
    ~LR_Problem();

    double loss(const Eigen::Ref<const ColVector>&);
    void gradient_Fx(const Eigen::Ref<const ColVector>&, Eigen::Ref<ColVector>);

protected:
    /** z are some reusable part of the processes */
    ColVector z_;
};

/// L1-Regularized Loss Logistic Regression
///
class L1R_LR_Problem : public LR_Problem
{

public:
    explicit L1R_LR_Problem(DatasetPtr, const std::vector<double>&);
    ~L1R_LR_Problem();

    void update_weights(Eigen::Ref<ColVector>, const Eigen::Ref<const ColVector>&,
                        const Eigen::Ref<const ColVector>&, const double&);

};
/// L2-Regularized Loss Logistic Regression
///
class L2R_LR_Problem : public LR_Problem
{

public:
    explicit L2R_LR_Problem(DatasetPtr, const std::vector<double>&);
    ~L2R_LR_Problem();
};

} // oplin

#endif// OPENLINEAR_FORMULA_H_
