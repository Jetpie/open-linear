// Linear Models
//
// Naming Convention: all mathematical matrix variables are denoted
//                    by captial letters like 'X' or 'W'
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory

#ifndef LINEAR_H_
#define LINEAR_H_

#include "solver.hpp"

#include <assert.h>

/// Dataset parameters
struct Dataset
{
    /** number of samples */
    size_t int n_sample;
    /** feature dimension */
    size_t dimension;
    /** targets */
    int* y;
    /** features w.r.t order of y*/
    CSR_Matrix X;

};
enum SolverType
{
    "L2R_LR",
    "L1R_LR";
};

/// Parameters for training
struct Parameter
{
    /** define the solver type for training model */
    int solver;
    /** tolerance for training stop criteria */
    double tolerance;
    /** C */
    double C;
    double bias;
};

/// Model Parameters
struct Model
{
    /** number of classes */
    size_t n_class;
    /** dimension of feature */
    size_t dimension;
    /** weights */
    CSR_Matrix W;
    /** define a bias, 0 if no bias setting */
    double bias;

};

bool check_dataset(const Dataset);
bool check_parameter(const Parameter);
bool check_model(const Model);

/// Base class for linear models
///
///
///
class LinearBase
{
protected:
    // symbolic links for short implementation views
    typedef shared_ptr<Model> ModelPtr;
    typedef shared_ptr<Parameter> ParamPtr;
    typedef shared_ptr<Dataset> DatasetPtr;

    //
    ModelPtr model_;
    //
    ParamPtr parameter_;
    void decision(void);

public:
    LinearBase(void) : model_(NULL) {};
    virtual ~LinearBase(void){};

    virtual void train(const DatasetPtr, const ParamPtr) = 0;
    virtual void predict_proba() = 0;
};

#endif //LINEAR_H_
