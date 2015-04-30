// Linear Models
//
// Naming Convention:
// all mathematical matrix variables are denoted by captial letters
// like 'X' or 'W'
//
// The key feature of this linear model is it was implemented by
// advanced and efficient c++ matrix library Eigen, but only for the
// training stage. After that, the weights are stored in normal c++
// array for fast predicting and robust interface.
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
#include <algorithm>
#include <Eigen/SparseCore>
#include <Eigen/Core>
using namespace std;
using namespace Eigen;

// Define Eigen vector and matrix types we will use
typedef Eigen::SparseMatrix<double, RowMajor> SpRowMatrix;
typedef Eigen::SparseMatrix<double, ColMajor> SpColMatrix;
typedef Eigen::SparseVector<double, RowMajor> SpRowVector;
typedef Eigen::SparseVector<double, ColMajor> SpColVector;
typedef Eigen::Matrix<double, Dynamic, 1      , ColMajor> ColVector;
typedef Eigen::Matrix<double, 1      , Dynamic, RowMajor> RowVector;
typedef Eigen::Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrix;
typedef Eigen::Matrix<double, Dynamic, Dynamic, ColMajor> ColMatrix;

// smart pointers
typedef std::shared_ptr<SpColMatrix> SpColMatrixPtr;
typedef std::shared_ptr<ColMatrix> ColMatrixPtr;

struct FeatureNode
{
    size_t i;
    double v;
};
/// Dataset parameters
struct Dataset
{
    /** number of samples */
    size_t n_samples;
    /** number of classes */
    size_t n_classes;
    /** feature dimension */
    size_t dimension;
    /** targets */
    vector<double> y;
    /** target labels */
    vector<double> labels;
    /**
     * features w.r.t order of y
     * dimension is dimension * n_samples
     */
    SpColMatrixPtr X;

};
enum FormulaType
{
    L2R_LR,
    L1R_LR
};
enum SolverType
{
    GD,
    SGD,
    NewtonCG,
    L_BFGS,
    TRON
};

/// Parameters for training
struct Parameter
{
    /** define the solver type for training model */
    int solver;
    /** define the objective function for solving */
    int formula;
    /** tolerance for training stop criteria */
    double tolerance;
    /** C */
    vector<double> C;
    double bias;
};

/// Model Parameters
///
/// An important note here is the key part of model - weights:
/// the representation of weight during training phase is an Eigen
/// dense matrix under consideration of the matrix manipultion
/// in fomulas. But finally converted to a normal c++ array as storage.
/// The consideration here is firstly, the low level c++ array is though
/// not safe in memory, I can validate the memory after training and
/// once the double* is stored, no other manipulation should have
/// previlige to modify it until it will be released by constructor.
/// All in all, the segmentation fault will not happen here under the
/// help of class encapsulation.
/// while secondly, the performance should have not much difference with
/// vector but slight better. The linear model always means fast in
/// industry and that "slightly" is good for all users.
///
struct Model
{
    /** number of classes */
    size_t n_classes;
    /** dimension of feature */
    size_t dimension;
    /** weights */
    double* W_;
    /** define a bias, 0 if no bias setting */
    double bias;
    /** labels of classes */
    vector<double> labels;
    // destructor must be called for double*
    ~Model()
    {
        delete [] W_;
    };

};

typedef vector<FeatureNode> FeatureVector;
// symbolic links for short implementation views
typedef shared_ptr<Model> ModelPtr;
typedef shared_ptr<Parameter> ParamPtr;
typedef shared_ptr<Dataset> DatasetPtr;


/// Base class for linear models
///
///
///
class LinearBase
{
protected:
    // model instance
    ModelPtr model_;
    bool trained_;
    // parameter instance
    ParamPtr parameter_;
    void predict_WTx(const FeatureVector, vector<double>&);

public:

    LinearBase(void);
    explicit LinearBase(const ModelPtr);
    void load_model(const ModelPtr);
    virtual ~LinearBase(void){};

    virtual bool is_trained();
    virtual size_t get_n_classes();
    virtual vector<double> get_labels();
    virtual void train(const DatasetPtr, const ParamPtr) = 0;
    virtual double predict(const FeatureVector);
    virtual double predict_proba(const FeatureVector, vector<double>&);

    void preprocess_data(const DatasetPtr, vector<size_t>&,
                         vector<size_t>&, vector<size_t>&);
};

#endif //LINEAR_H_
