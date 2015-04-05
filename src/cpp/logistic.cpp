// Logistic Regression
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory

#include "logistic.hpp"

// LogisticRegression::LogisticRegression() {}

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
    if(!param)
    {
        cerr << "LogisticRegression::train : Error input, param NULL or not valid"
             << __FILE__ << "," << __LINE__ << endl;
    }
    if(!dataset)
    {
        cerr << "LogisticRegression::train : Error input, dataset NULL or not valid"
             << __FILE__ << "," << __LINE__ << endl;
    }

    size_t n_sample = dataset->n_samples;
    size_t dimension = dataset->dimension;

    // initialize model
    ModelPtr model = make_shared<Model>();
    // sanity check
    assert(model);

    model->bias = param->bias;

}
