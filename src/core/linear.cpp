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
#include "linear.hpp"
#include <fstream>

namespace oplin{
using std::cout;
using std::cerr;
using std::endl;
using namespace Eigen;
void VOUT(const char* fmt, ...)
{
#ifdef _OPLIN_DEBUG_
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
#endif
}
LinearBase::LinearBase() : model_(nullptr),trained_(false) {};
LinearBase::LinearBase(ModelUniPtr model)
{
    load_model( std::move(model) );
}

/**
 * To load a model from outside. The model will be validated
 *
 * @param model shared_ptr to Model instance
 */
void
LinearBase::load_model(ModelUniPtr model)
{
    // TODO: add check of model
    this->model_.reset(model.release());
    this->trained_ = true;
}
/**
 * Export model (unique_ptr) to outside. As long as the control was taken
 * by outside, the inner model should be retrained or re-taken for the
 * safty consideration.
 *
 * @return unique_ptr for Model
 */
ModelUniPtr
LinearBase::export_model()
{
    this->trained_ = false;
    return std::move(model_);
}
void
LinearBase::export_model_to_file(const std::string& filename)
{
    std::ofstream outfile(filename,std::ios::out);
    outfile.precision(10);
    // this sort of error should never happen?
    if(!outfile.is_open())
    {
        cerr << "LinearBase::export_model_to_file : Could not open output file!"
             << __FILE__ << "," << __LINE__ << endl;
        throw std::runtime_error("Could not open outpuf file!");
    }
    outfile << "n_classes " << model_->n_classes <<"\n";
    outfile << "labels ";
    for(std::vector<double>::iterator it(model_->labels.begin());it!=model_->labels.end();++it)
        outfile << *it <<" ";
    outfile << "\n";
    outfile << "dimension " << model_->dimension <<"\n";
    outfile << "bias " << model_->bias <<"\n";
    // output weights
    // for binary classification only one weights trained
    size_t n_ws = model_->n_classes == 2 ? 1 : model_->n_classes;
    outfile << "weights\n";
    double* cur_w = model_->W_;
    size_t i,j;
    for(i = 0; i < model_->dimension ; ++i)
    {
        for(j=0; j<n_ws; ++j)
        {
            outfile << *cur_w <<" ";
            ++cur_w;
        }
        outfile << "\n";
    }
}
/**
 * Return a state if the model parameter is trained
 *
 * @return true if the model is trained else false
 */
bool
LinearBase::is_trained()
{
    return this->trained_;
}
/**
 * Getter for n_classes
 *
 * @return number of classes of model
 */
size_t
LinearBase::get_n_classes()
{
    return this->model_->n_classes;
}
/**
 * Getter for labels
 *
 * @return vector of labels
 */
std::vector<double>
LinearBase::get_labels()
{
    return this->model_->labels;
}

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
LinearBase::preprocess_data(const DatasetPtr dataset, std::vector<size_t>& count,
                            std::vector<size_t>& start_idx, std::vector<size_t>& perm_idx)
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

    // class index for each sample (0...n_samples)
    std::vector<size_t> index;
    index.reserve(n_samples);
    for(i=0; i < n_samples; ++i)
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
            throw(std::out_of_range("label out of range"));
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

/**
 * Compute the intermedient result of linear combination with input x
 * and model weights WTx
 *
 * @param x   sparse storage of feature node of <k,v> pair
 * @param WTx matrix meaning W transpose by x
 *
 * @return label of current prediction
 */
void
LinearBase::predict_WTx(FeatureVector x, std::vector<double>& WTx)
{
    // no WTx valdation is need as this method is encapsulated to outside.
    //
    size_t n_classes = model_->n_classes;
    // for binary classification only one weights trained
    size_t n_ws = n_classes == 2 ? 1 : n_classes;

    double* w = model_->W_;
    size_t i;
    // weights for current feature dimension
    double* cur_w;
    // compute W^T x
    {
        for(size_t n = 0; n<x.size(); ++n)
        {
            cur_w = &w[(x[n].i) *n_ws];
            for(i=0; i<n_ws; ++i)
            {
                WTx[i] += x[n].v * (*cur_w);
                ++cur_w;
            }
        }
    }
    // if bias term are applied
    if(model_->bias_values_)
    {
        for(i=0; i<n_ws; ++i)
        {
            WTx[i] += model_->bias_values_[i];
        }
    }
    return;

}

/**
 * Make prediction on label of current input x.
 *
 * @param x sparse storage of feature node of <k,v> pair
 *
 * @return label of current prediction
 */
double
LinearBase::predict(FeatureVector x)
{
    // initialize WTx by size
    std::vector<double> WTx(model_->n_classes);
    predict_WTx(x, WTx);
    size_t i;
    // threshold 0 for binary classification
    if(model_->n_classes==2)
        return WTx[0] > 0 ? model_->labels[0] : model_->labels[1];
    else
    {
        size_t best_idx = 0;
        for(i=0;i<model_->n_classes;++i)
        {
            if(WTx[i] > WTx[best_idx])
                best_idx = i;
        }
        return model_->labels[best_idx];
    }

}

/**
 * Predict the label with probability if this model can be paraphrase
 * as a probability view.
 *
 * @param x           sparse storage of feature node of <k,v> pair
 * @param probability respected probabilities of labels
 *
 * @return label of current prediction
 */
double
LinearBase::predict_proba(FeatureVector x, std::vector<double>& probability)
{

    // use the swap trick here. will validate the benchmark later
    // TODO: validate swap trick here
    std::vector<double>(model_->n_classes).swap(probability);
    predict_WTx(x, probability);

    // initialize label
    double label = 0;
    size_t i;
    if(model_->n_classes==2)
    {
        label = probability[0] > 0 ? model_->labels[0] : model_->labels[1];

        probability[0] = 1 / (1 + exp(-probability[0]));
        probability[1] = 1 - probability[0];
    }
    else
    {
        size_t best_idx = 0;
        double sum = 0;
        for(i=0;i<model_->n_classes;++i)
        {
            probability[i] = 1 / (1 + exp(-probability[i]));
            sum += probability[i];
            if(probability[i] > probability[best_idx])
                best_idx = i;
        }
        // sum normalize
        for(i = 0; i < model_->n_classes; ++i)
            probability[i] = probability[i] / sum;

        label = model_->labels[best_idx];
    }

    return label;
}

} // oplin
