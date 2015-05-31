// Command line interface for train settings
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#include <stdio.h>     /* for printf */
#include <stdlib.h>    /* for exit */
#include <getopt.h>
#include "linear.hpp"
#include "logistic.hpp"
#include "high_level_function.hpp"

using std::cout;
using std::cerr;
using std::endl;
using namespace oplin;

void print_help()
{
    cerr
    << "Usage: train [train options] dataset_file model_file " << endl
    << "train options:" << endl
    << "-s [--solver]: solver type (default 0)" <<endl
    << "\t0 -- Steepest(Gradient) Descent" <<endl
    << "\t1 -- Stochastic Gradient Descent" <<endl
    << "\t2 -- L-BFGS" <<endl
    << "-p [--problem]: problem type (default 0)" <<endl
    << "\t0 -- L1-regularized logistic regression" <<endl
    << "\t1 -- L2-regularized logistic regression" << endl
    << "-b [--bias]: bias term, -1 for no bias term applied (default -1)" <<endl
    << "-r [--rela_tol]: relative tolerance between two epochs (default 1e-5)" << endl
    << "-a [--abs_tol]: absolute tolerance of loss (default 0.1)" << endl
    << "-m [--max_epoch]: max epoch setting (default 500)" << endl
    << "-d [--dimension]: maximum dimension of feature exclude bias term (no default, must be specified)" << endl
    << "-e [--estimate_n_samples]: estimation on number of training samples."
        " accurate estimation can improve the memory usage (default 1000)" << endl
    << "-l [--learning_rate]: learning rate setting (default 0.01)"
    <<endl;
}

int main(int argc, char **argv)
{
    ParamPtr param = std::make_shared<Parameter>();
    param->solver_type = 0;
    param->problem_type = 0;
    param->rela_tol = 1e-5;
    param->abs_tol = 0.1;
    param->max_epoch = 500;
    param->learning_rate = 0.01;
    // std::vector<double> C(dataset->n_samples,1.);
    // param->C = C;

    int bias = -1;
    size_t n_features = 0, estimate_n_samples = 1000;
    struct option long_options[] = {
        {"solver",   required_argument, 0,  's' },
        {"problem",  required_argument, 0,  'p' },
        {"bias",     required_argument, 0,  'b' },
        {"rela_tol", required_argument, 0,  'r' },
        {"abs_tol",  required_argument, 0,  'a' },
        {"max_epoch",required_argument, 0,  'm' },
        {"learning_rate",required_argument, 0,  'l' },
        {"dimension",required_argument, 0,  'd' },
        {"estimate_samples",required_argument, 0,  'e' },
        {"help",     no_argument,       0,  'h' },
        {0,0,0,0}
    };

    int opt,option_index = 0;
    while ((opt = getopt_long(argc, argv, "s:p:hb:r:a:m:l:d:e:",
                              long_options, &option_index)) != -1) {
        switch (opt) {
        case 's':
            param->solver_type = atoi(optarg);
            break;
        case 'p':
            param->problem_type = atoi(optarg);
            break;
        case 'b':
            bias = atof(optarg);
            break;
        case 'r':
            param->rela_tol = atof(optarg);
            break;
        case 'a':
            param->abs_tol = atof(optarg);
            break;
        case 'm':
            param->max_epoch = atoi(optarg);
            break;
        case 'l':
            param->learning_rate = atof(optarg);
            break;
        case 'd':
            n_features = atoi(optarg);
            break;
        case 'e':
            estimate_n_samples = atoi(optarg);
            break;
        case 'h':
            print_help();
            return EXIT_SUCCESS;
        case '?':
            print_help();
            return EXIT_FAILURE;
        default: /* '?' */
            print_help();
            return EXIT_FAILURE;
        }
    }

    // +2 for train sample file and model file
    if (optind + 2 != argc || !n_features )
    {
        print_help();
        return EXIT_FAILURE;
    }

    //
    std::string sample_file(argv[optind++]);
    std::string model_file(argv[optind++]);
    cout << "input sample file : " << sample_file << endl;
    cout << "output model file : " << model_file << endl;

    // set std::cout precision
    std::cout.precision(10);
    // read dataset
    DatasetPtr dataset = read_dataset(sample_file, n_features , bias, estimate_n_samples);
    // set temp soft margin for temporal
    std::vector<double> C(dataset->n_samples,1.);
    param->C = C;
    // logistic regresion instance
    std::shared_ptr<LinearBase> lr= std::make_shared<LogisticRegression>();
    // train model
    lr->train(dataset, param);
    // predict file
    predict_all(sample_file,model_file,lr,true);


    return EXIT_SUCCESS;
}
