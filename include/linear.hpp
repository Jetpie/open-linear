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

namespace oplin{

// Define Eigen vector and matrix types we will use
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpRowMatrix;
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpColMatrix;
typedef Eigen::SparseVector<double, Eigen::RowMajor> SpRowVector;
typedef Eigen::SparseVector<double, Eigen::ColMajor> SpColVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1      , Eigen::ColMajor> ColVector;
typedef Eigen::Matrix<double, 1      , Eigen::Dynamic, Eigen::RowMajor> RowVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> ColMatrix;

// smart pointers
typedef std::shared_ptr<SpColMatrix> SpColMatrixPtr;
typedef std::shared_ptr<ColMatrix> ColMatrixPtr;

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
    std::vector<double> y;
    /** target labels */
    std::vector<double> labels;
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
    int solver_type;
    /** define the objective function for solving */
    int problem_type;
    /**
     * relative tolerance between two continuous iterations
     * for training stop criteria
     */
    double rela_tol;
    /** absolute loss tolerance for iterative method */
    double abs_tol;
    /** learning rate for iterative method */
    double lambda;
    /** maximum iteration for iterative method */
    size_t max_epoch;
    /** C */
    std::vector<double> C;
    double bias;
};
typedef std::shared_ptr<Parameter> ParamPtr;
typedef std::shared_ptr<Dataset> DatasetPtr;

struct FeatureNode
{
    size_t i;
    double v;
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
    std::vector<double> labels;
    // destructor must be called for double*
    ~Model()
    {
        delete [] W_;
    };

};

typedef std::vector<FeatureNode> FeatureVector;
// symbolic links for short implementation views
typedef std::shared_ptr<Model> ModelPtr;


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
    void predict_WTx(const FeatureVector, std::vector<double>&);

public:

    LinearBase(void);
    explicit LinearBase(const ModelPtr);
    void load_model(const ModelPtr);
    virtual ~LinearBase(void){};

    virtual bool is_trained();
    virtual size_t get_n_classes();
    virtual std::vector<double> get_labels();
    virtual void train(const DatasetPtr, const ParamPtr) = 0;
    virtual double predict(const FeatureVector);
    virtual double predict_proba(const FeatureVector, std::vector<double>&);

    void preprocess_data(const DatasetPtr, std::vector<size_t>&,
                         std::vector<size_t>&, std::vector<size_t>&);
};

} // oplin
// using namespace oplin;
// using namespace oplin::model;
#endif //LINEAR_H_
