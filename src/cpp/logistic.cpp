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
 * @param dataset training dataset
 * @param params parameters
 *
 */
void
LogisticRegression::train(const DatasetPtr dataset, const ParamPtr param)
{
    // input sanity check
    if(!param)
        cerr << "LogisticRegression::train : Error input, param NULL or not valid"
             << __FILE__ << "," << __LINE__ << endl;
    if(!dataset)
        cerr << "LogisticRegression::train : Error input, dataset NULL or not valid"
             << __FILE__ << "," << __LINE__ << endl;

    size_t n_samples = dataset->n_samples;
    size_t dimension = dataset->dimension;
    size_t n_classes = dataset->n_classes;

    // initialize model
    ModelPtr model = make_shared<Model>();
    // sanity check
    assert(model);

    model->bias = param->bias;
    model->dimension = dimension;
    model->labels = dataset->labels;
    model->n_classes = n_classes;

    vector<size_t> count;
    vector<size_t> start_idx;
    vector<size_t> perm_idx;
    count.reserve(n_classes);
    start_idx.reserve(n_classes);
    perm_idx.reserve(n_samples);

    // preprocess the training dataset
    preprocess_data(dataset, count, start_idx, perm_idx);
    for(size_t i=0;i<n_samples;++i)
    {
        cout << perm_idx[i] << ",";
    }
    cout << endl;

}
void
LogisticRegression::predict_proba()
{

}
