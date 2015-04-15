// Linear Models
//
// Naming Convention:
// all mathematical matrix variables are denoted by captial letters
// like 'X' or 'W'
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#include "linear.hpp"

/**
 * Preprocess the dataset.
 *
 * @param dataset   the dataset pointer
 * @param count     count of each class
 * @param start_idx start index of each class in the rearrange group
 * @param perm_idx  permutation from index of each element
 *
 */
void
LinearBase::preprocess_data(const DatasetPtr dataset, vector<size_t>& count,
                            vector<size_t>& start_idx, vector<size_t>& perm_idx)
{
    size_t n_samples = dataset->n_samples;
    size_t n_classes = dataset->n_classes;
    // check if perm_idx and start are empty size
    if(!perm_idx.empty())
    {
        perm_idx.clear();
        perm_idx.reserve(n_samples);
    }
    if(!start_idx.empty())
    {
        start_idx.clear();
        start_idx.reserve(n_classes);
    }
    size_t i=0,j=0;
    // set all counts to 0s
    count.assign(n_classes,0);

    // index for each class target (0...n)
    vector<size_t> index;
    index.reserve(n_samples);
    for(size_t i=0; i < n_samples; ++i)
    {
        double label = dataset->y[i];
        for(j=0; j<n_classes; ++j)
        {
            if(label == dataset->labels[j])
            {
                ++count[j];
                break;
            }
        }
        if(j==n_classes)
        {
            cerr << "LinearBase::preprocess_data : Dataset label error!"
                 << __FILE__ << "," << __LINE__ << endl;
        }
        index.push_back(j);
    }
    // start from 0 index
    start_idx.push_back(0);
    for(i=1; i<n_classes; ++i)
        start_idx.push_back(start_idx[i-1] + count[i-1]);

    // set permute from index of each element
    for(i=0; i< n_samples; ++i)
        perm_idx[start_idx[index[i]]++] = i;

    // reset start index
    start_idx[0] = 0;
    for(i=1; i<n_classes; ++i)
        start_idx[i] = start_idx[i-1] + count[i-1];

    return;
}

double
LinearBase::predict(const SpColVector x)
{
    size_t n_classes = model_->n_classes;
    size_t dimension = model_->dimension;
    // validate dimension
    if(x.rows() != dimension)
    {
        cerr << "LinearBase::predict : input dimension not valid!"
             << __FILE__ << "," << __LINE__ << endl;
    }

    ColVector WTx = (*model_->W).transpose() * x;

    if(n_classes == 2)
    {
        return (WTx(0) > 0)? model_->labels[0] : model_->labels[1];
    }
    else
    {
        MatrixXd::Index max_i;
        WTx.maxCoeff(&max_i);

        return model_->labels[max_i];
    }
}

double
LinearBase::predict_proba(const SpColVector x, vector<double> probability)
{
    size_t n_classes = model_->n_classes;
    size_t dimension = model_->dimension;
    // validate dimension
    if(x.rows() != dimension)
    {
        cerr << "LinearBase::predict : input dimension not valid!"
             << __FILE__ << "," << __LINE__ << endl;
    }

    ColVector WTx = (*model_->W).transpose() * x;

    if(n_classes == 2)
    {
        return (WTx(0) > 0)? model_->labels[0] : model_->labels[1];
    }
    else
    {
        MatrixXd::Index max_i;
        WTx.maxCoeff(&max_i);

        return model_->labels[max_i];
    }
}
