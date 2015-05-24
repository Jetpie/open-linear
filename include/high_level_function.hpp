// Command interfaces
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#ifndef HIGH_LEVEL_FUNCTION_H_
#define HIGH_LEVEL_FUNCTION_H_

#include "linear.hpp"

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
#include <map>
/**
 * Print help information on stdout
 *
 */
void print_help()
{
    cout << "HELP" <<endl;
}

/**
 * Parse command line arguments
 *
 *
 */
void parse_command_line(int argc, char** argv)
{

}

/**
 * Read dataset from file. An estimation of number of instance is better
 * to be given for better memory usage.
 *
 * @param filename  input file name of dataset
 * @param n_entries estimated number of entries of datasets. Will run as
 *                  normal though not accurate but may cause memory error
 *                  because no enough space is reserved.
 * @param dimension dimension of feature space
 *
 * @return shared_ptr to loaded dataset
 */
DatasetPtr
read_dataset(const string filename, const size_t dimension , const size_t n_entries = 1000)
{
    //
    size_t n_samples = 0;
    vector<double> y;
    y.reserve(n_entries);

    // vector to store Triplets to construct sparse matrix
    typedef Eigen::Triplet<double> Tri;
    std::vector<Tri> triplets;
    triplets.reserve(n_entries);

    // read through file;
    ifstream infile(filename);
    std::set<double> classes;
    for(string line; std::getline(infile,line);)
    {
        stringstream ss(line);
        string item;
        std::getline(ss,item,' ');
        double label = std::stod(item);
        y.push_back(label);
        // check if the target has been counted
        if(classes.find(label) == classes.end())
            classes.insert(label);
        while(std::getline(ss,item,' '))
        {
            int i;
            float v_ij;
            sscanf(item.c_str(),"%d:%f",&i,&v_ij);
            // j = n_samples;
            triplets.push_back(Tri(i-1,n_samples,v_ij));
        }
        ++n_samples;
    }
    infile.close();
    // assign dataset model
    DatasetPtr dataset = make_shared<Dataset>();
    if(!dataset)
    {
        cerr << "read_dataset : DatasetPtr allocation failed!"
             << __FILE__ << "," << __LINE__ << endl;
    }
    dataset->n_samples = n_samples;
    dataset->n_classes = classes.size();
    dataset->dimension = dimension;
    dataset->labels = vector<double>(classes.begin(),classes.end());
    dataset->y = y;
    dataset->X = make_shared<SpColMatrix>(dimension,n_samples);
    if(!dataset->X)
    {
        cerr << "read_dataset : SpColMatrixPtr allocation failed!"
             << __FILE__ << "," << __LINE__ << endl;
    }

    // assign values
    (dataset->X)->setFromTriplets(triplets.begin(),triplets.end());

    return dataset;
}

/*
 * Load model from file. Fixed terms for each line
 *
 * @param filename inpute filename for model
 *
 * @return shared_ptr for Model
 */
ModelPtr load_model(const string& filename)
{
    ModelPtr model = make_shared<Model>();
    // sanity check
    if(!model)
    {
        cerr << "read_dataset : ModelPtr allocation failed!"
             << __FILE__ << "," << __LINE__ << endl;
    }

    ifstream infile(filename);

    for(string line; std::getline(infile,line);)
    {
        stringstream ss(line);
        string item;
        // get content indicator
        std::getline(ss,item,' ');

        if(item == "solver_type")
        {
            cout << "solver_type meet, jump over" << endl;
        }
        else if(item == "nr_class")
        {
            std::getline(ss,item,' ');
            model->n_classes = std::stod(item);
        }
        else if(item == "label")
        {
            model->labels.reserve(model->n_classes);
            while( std::getline(ss,item,' ') )
                model->labels.push_back(std::stod(item));
        }
        else if(item == "nr_feature")
        {
            std::getline(ss,item,' ');
            model->dimension = std::stod(item);
        }
        else if(item == "bias")
        {
            std::getline(ss,item,' ');
            model->bias = std::stod(item);
        }
        else if(item == "w")
        {
            size_t cols;
            if(model->n_classes == 2)
                cols = 1;
            else
                cols = model->n_classes;
            double* W_ = new double[cols * model->dimension];
            // sanity check
            if(!W_)
            {
                cerr << "read_dataset : ModelPtr allocation failed!"
                     << __FILE__ << "," << __LINE__ << endl;
            }
            for(size_t i =0;i < model->dimension; ++i)
            {
                std::getline(infile,line);
                stringstream ss(line);
                string item;
                for(size_t j =0; j < cols; ++j)
                {
                    std::getline(ss,item,' ');
                    W_[i*cols + j] = std::stod(item);
                }
            }
            model->W_ = W_;

            if(std::getline(infile,line))
            {
                cerr << "Model error, please check" << endl;
            }
        }
        else
        {
            cerr << "Unexpected words in model file : " << item << endl;
            cerr << "Please check model and read again..."  << endl;
            return NULL;
        }
    }
    infile.close();
    return model;

}

/**
 * Make prediction on all label and feature pairs of the input file.
 * If the label is not valid in the model, the predictor will jump over.
 * A confusion matrix can be recorded for evaluation.
 *
 * @param input            input file path
 * @param output           output file path
 * @param lb               shared_ptr to a linear model
 * @param flag_probability if print out the probabilities
 * @param estimate_n       estimation of number of input feature.
 *                         this will automatically be recomputed if exceeds.
 */
void predict_all(string& input, string& output, shared_ptr<LinearBase> lb,
                 bool flag_probability = false, size_t estimate_n = 100)
{
    if(!lb->is_trained())
    {
        cerr << "predict_all : Model not trained,  please train the model first!"
             << __FILE__ << "," << __LINE__ << endl;
    }
    ifstream infile(input);
    ofstream outfile(output,ios::out);
    if(!outfile.is_open())
    {
        cerr << "predict_all : output file open error!"
             << __FILE__ << "," << __LINE__ << endl;

    }

    size_t n_classes = lb->get_n_classes();
    const vector<double> labels = lb->get_labels();
    // first line record all predictable labels
    std::map<double,size_t> label_index;
    outfile << "labels";
    for(size_t i = 0;i < labels.size();++i)
    {
        label_index[labels[i]] = i;
        outfile << " " << labels[i];
    }
    outfile << "\n";
    // initialize confusion matrix
    vector<vector<size_t> > confusion_matrix(n_classes,vector<size_t>(n_classes));

    // initialize iterator buffer to check if input label is valid
    map<double,size_t>::iterator it = label_index.end();

    // buffers for label and index
    double true_label, pred_label;
    size_t true_i,pred_i;

    double n_correct = 0, n_samples = 0;

    FeatureVector x;
    size_t max_n_feature = 0;

    // read through file
    for(string line; std::getline(infile,line);)
    {
        stringstream ss(line);
        string item;
        std::getline(ss,item,' ');
        true_label = std::stod(item);
        it = label_index.find(true_label);

        // if true label is not known to model
        if(it == label_index.end())
        {
            cout << "Warning: Unknown label(" << true_label
                 << ") to model in file, jump over!";
            // jump over
            continue;
        }
        true_i = it->second;

        // flush the feature buffer
        max_n_feature = 0;
        x.clear();
        while(std::getline(ss,item,' '))
        {
            int i;
            float v_ij;
            sscanf(item.c_str(),"%d:%f",&i,&v_ij);
            max_n_feature++;
            if(max_n_feature > estimate_n)
            {
                estimate_n *= 2;
                x.reserve(estimate_n);
            }
            x.push_back({(size_t) (i-1), v_ij});
        }

        // TODO: add check if model is a probability model
        if(flag_probability)
        {
            vector<double> p;
            pred_label = lb->predict_proba(x,p);
            outfile << pred_label;
            for(size_t i = 0; i < n_classes;++i)
                outfile << " " << p[i];
            outfile << "\n";
        }
        else
        {
            pred_label = lb->predict(x);
            outfile << pred_label << "\n";
        }

        // the labels is get from model, no safty problem
        pred_i = label_index[pred_label];

        // +1 for confusion matrix
        ++confusion_matrix[pred_i][true_i];
        // cout << "pred_i " << pred_i <<" true_i " << true_i << endl;
        if(true_i == pred_i)
            ++n_correct;
        ++n_samples;
    }
    infile.close();
    outfile.close();

    double accuracy = (n_correct / n_samples) * 100;
    printf("Prediction Accuracy : %.4f%%\n",accuracy);

    return;
}

#endif //HIGH_LEVEL_FUNCTION_H_
