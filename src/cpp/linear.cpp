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

size_t
LinearBase::preprocess_data(const DatasetPtr dataset, vector<double>& labels,
                            vector<size_t>& count, vector<size_t>& perm_idx)
{
    size_t n_samples = dataset->n_samples;
    size_t n_classes = 0;
    // index for each class target (0...n)
    vector<size_t> index;
    index.reserve(n_samples);
    for(size_t i=0; i < n_samples; ++i)
    {
        double label = dataset->y[i];
        size_t j;
        for(j=0; j<n_classes; ++j)
        {
            if(label == labels[j])
            {
                ++count[j];
                break;
            }
        }
        index.push_back(j);
        if(j == n_classes)
        {
            labels.push_back(label);
            count.push_back(1);
            ++n_classes;
        }
    }



    return n_classes;
}
