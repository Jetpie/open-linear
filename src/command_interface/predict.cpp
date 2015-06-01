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
    cout
    << "Usage: predict [predict options] dataset_file model_file outpuf_file" << endl
    << "predict options:" << endl
    << "-p [--probability]: output the probability or not, 0 or 1 (default 0)" <<endl
    << "-e [--estimate_n_samples]: estimation on number of training samples."
        " accurate estimation can improve the memory usage (default 100)" << endl
    << "-h [--help]: print usage help information"
    <<endl;
}

int main(int argc, char **argv)
{

    int probability = 0,estimate_n_samples = 100;
    struct option long_options[] = {
        {"probability",   required_argument, 0,  'p' },
        {"estimate_samples",required_argument, 0,  'e' },
        {"help",     no_argument,       0,  'h' },
        {0,0,0,0}
    };

    int opt,option_index = 0;
    while ((opt = getopt_long(argc, argv, "p:e:h",
                              long_options, &option_index)) != -1) {
        switch (opt) {
        case 'p':
            probability = atoi(optarg);
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
    if (optind + 3 != argc)
    {
        print_help();
        return EXIT_FAILURE;
    }

    //
    std::string sample_file(argv[optind++]);
    std::string model_file(argv[optind++]);
    std::string output_file(argv[optind++]);
    cout << "input sample file : " << sample_file << endl;
    cout << "model file : " << model_file << endl;
    cout << "output file : " << output_file << endl;
    std::shared_ptr<LinearBase> lr= std::make_shared<LogisticRegression>();
    lr->load_model(std::move(load_model(model_file)));
    predict_all(sample_file,output_file, lr, probability, estimate_n_samples);


    return EXIT_SUCCESS;
}
