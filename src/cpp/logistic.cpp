// Logistic Regression
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory

#include "logistic.hpp";

LogisticRegression::LogisticRegression() : LinearBase() {}

/**
 * Train the coefficients using depends on the parameters.
 *
 * @param dataset datasets
 * @param params parameters
 *
 */
void
LogisticRegression::train(const DatasetPtr dataset, const ParamPtr param)
{
    // input check
    if(!param || !check_parameter(param))
    {
        cerr << "LogisticRegression::train : Error input, param NULL or not valid"
             << __FILE__ << "," << __LINE__ << endl;
    }
    if(!dataset || !check_dataset(dataset))
    {
        cerr << "LogisticRegression::train : Error input, dataset NULL or not valid"
             << __FILE__ << "," << __LINE__ << endl;
    }

    int n_sample = dataset->n_sample;
    int dimension = dataset->dimension;

    // initialize model
    Model* model = make_shared<Model>();
    // sanity check
    assert(model);
    model->bias = param->bias;

}
