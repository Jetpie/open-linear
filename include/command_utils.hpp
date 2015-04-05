// Command interfaces
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#ifndef COMMAND_UTILS_H_
#define COMMAND_UTILS_H_

#include "linear.hpp"

#include <stdio.h>
#include <string>
#include <fstream>
#include <sstream>
#include <set>
void print_help()
{
    cout << "HELP" <<endl;
}

void parse_command_line(int argc, char** argv)
{

}

/**
 * Read dataset from file. An estimation of number of instance is better
 * to be given for better memory usage.
 *
 * @param filename
 * @param n_entries
 * @param dimension
 *
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

    // read throgh file;
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
        cerr << "read_dataset : P_CRS_ColMat allocation failed!"
             << __FILE__ << "," << __LINE__ << endl;
    }

    // assign values
    (dataset->X)->setFromTriplets(triplets.begin(),triplets.end());

    return dataset;
}
#endif //COMMAND_UTILS_H_
