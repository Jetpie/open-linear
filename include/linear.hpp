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

#include <assert.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <Eigen/SparseCore>
#include <Eigen/Core>
using namespace std;
using namespace Eigen;

// Define Eigen vector and matrix types we will use
typedef Eigen::SparseMatrix<double, RowMajor> CRS_RowMatrix;
typedef Eigen::SparseMatrix<double, ColMajor> CRS_ColMatrix;
typedef Eigen::SparseVector<double, RowMajor> CRS_RowVector;
typedef Eigen::SparseVector<double, ColMajor> CRS_ColVector;
typedef Eigen::Matrix<double, Dynamic, 1      , ColMajor> ColVector;
typedef Eigen::Matrix<double, 1      , Dynamic, RowMajor> RowVector;
typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrix;
typedef Eigen::Matrix<double, Dynamic, Dynamic, ColMajor> ColMatrix;

// smart pointers
typedef std::shared_ptr<CRS_ColMatrix> P_CRS_ColMat;

/// Dataset parameters
struct Dataset
{
    /** number of samples */
    size_t n_samples;
    /** feature dimension */
    size_t dimension;
    /** targets */
    vector<double> y;
    /**
     * features w.r.t order of y
     * dimension is dimension * n_samples
     */
    P_CRS_ColMat X;

};
enum SolverType
{
    L2R_LR,
    L1R_LR
};

/// Parameters for training
struct Parameter
{
    /** define the solver type for training model */
    int solver;
    /** tolerance for training stop criteria */
    double tolerance;
    /** C */
    vector<double> C;
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
    ColMatrix W;
    /** define a bias, 0 if no bias setting */
    double bias;
    /** labels of classes */
    vector<int> labels;

};

// symbolic links for short implementation views
typedef shared_ptr<Model> ModelPtr;
typedef shared_ptr<Parameter> ParamPtr;
typedef shared_ptr<Dataset> DatasetPtr;

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
    // model instance
    ModelPtr model_;
    // parameter instance
    ParamPtr parameter_;
    void decision(void);

public:
    LinearBase(void) : model_(NULL) {};
    virtual ~LinearBase(void){};

    virtual void train(const DatasetPtr, const ParamPtr) = 0;
    virtual void predict_proba() = 0;
};

#endif //LINEAR_H_
