// Linear Models
//
// Special Naming Convention:
//
// All mathematical matrix variables are named by captial letters
// (e.g. X means input feature dataset). Vectors are named by lowercase
// (e.g. w means weights vector). These variables are always from Eigen.
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

#ifndef OPENLINEAR_LINEAR_H_
#define OPENLINEAR_LINEAR_H_

#include <assert.h>
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <exception>
#include <stdexcept>
#include <Eigen/SparseCore>
#include <Eigen/Core>
#include <stdio.h>
#include <stdarg.h>
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
typedef std::shared_ptr<ColVector> ColVectorPtr;

//
void VOUT(const char* fmt, ...);
// very very light-weight and useful structure for <key,value> pair storage
template <class K, class V>
struct KeyValue
{
    K i;
    V v;
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
    std::vector<double> y;
    /** target labels */
    std::vector<double> labels;
    /**
     * features w.r.t order of y
     * dimension is dimension * n_samples
     */
    SpColMatrixPtr X;
    double bias;
    Dataset() : bias(-1.){}

};
enum FormulaType
{
    L1R_LR,
    L2R_LR

};
enum SolverType
{
    GD,
    SGD,
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
    double learning_rate;
    /** maximum iteration for iterative method */
    size_t max_epoch;
    /** C */
    // std::vector<double> C;
    double base_C;
    std::vector<KeyValue<double,double> > adjust_C;

    Parameter() : solver_type(0.), problem_type(0.){}
};
typedef std::shared_ptr<Parameter> ParamPtr;
typedef std::shared_ptr<Dataset> DatasetPtr;


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
///
/// Secondly, the performance should have not much difference with
/// vector but slightly better. The linear model always means fast in
/// industry and that "slightly" is good for all users.
///
struct Model
{
    /** number of classes */
    size_t n_classes;
    /** dimension of feature */
    size_t dimension;
    /** define a bias, 0 if no bias setting */
    double bias;
    /** labels of classes */
    std::vector<double> labels;
    Model() : W_(NULL),bias_values_(NULL){}
    // destructor must be called for double*
    ~Model()
    {
        delete [] W_;
        delete [] bias_values_;
    }
    void set_bias_values(double* vals)
    {
        if(bias_values_)
            delete [] bias_values_;
        bias_values_ = vals;
    }
    void set_weights(double* w)
    {
        if(W_)
            delete [] W_;
        W_ = w;
    }
// I encapsulate the two pointers to avoid wrong reference in productive env.
private:
    double* W_;
    double* bias_values_;
    /** weights */

    friend class LinearBase;
};

typedef std::vector<KeyValue<size_t,double> > FeatureVector;
// symbolic links for short implementation views
// typedef std::shared_ptr<Model> ModelPtr;

// the current design keep model to only one owner to avoid wrong reference in
// productive env.
typedef std::unique_ptr<Model> ModelUniPtr;


/// Base class for linear models
///
///
///
class LinearBase
{
protected:
    // model instance
    ModelUniPtr model_;
    bool trained_;
    void predict_WTx(const FeatureVector, std::vector<double>&);
    void preprocess_data(const DatasetPtr, std::vector<size_t>&,
                         std::vector<size_t>&, std::vector<size_t>&);
public:

    LinearBase(void);
    explicit LinearBase(ModelUniPtr);
    virtual ~LinearBase(void){};

    virtual bool is_trained();
    virtual size_t get_n_classes();
    virtual std::vector<double> get_labels();
    virtual void load_model(ModelUniPtr);
    virtual ModelUniPtr export_model();
    virtual void export_model_to_file(const std::string&);
    virtual void train(const DatasetPtr, const ParamPtr) = 0;
    virtual double predict(const FeatureVector);
    virtual double predict_proba(const FeatureVector, std::vector<double>&);
};

} // oplin
// using namespace oplin;
// using namespace oplin::model;
#endif //OPENLINEAR_LINEAR_H_
