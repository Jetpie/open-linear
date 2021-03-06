// Command line interface for train settings
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#include <getopt.h>
#include "logistic.hpp"
#include "high_level_function.hpp"

using std::cout;
using std::cerr;
using std::endl;


void print_help()
{
    cout
    << "Usage: train [train options] dataset_file model_file " << endl
    << "train options:" << endl
    << "-s [--solver]: Solver type (default 0)" <<endl
    << "\t0 -- Steepest(Gradient) Descent" <<endl
    << "\t1 -- Stochastic Gradient Descent" <<endl
    << "\t2 -- L-BFGS" <<endl
    << "-p [--problem]: Problem type (default 0)" <<endl
    << "\t0 -- L1-regularized logistic regression" <<endl
    << "\t1 -- L2-regularized logistic regression" << endl
    << "-b [--bias]: Bias term, -1 for no bias term applied (default -1)" <<endl
    << "-r [--rela_tol]: Relative tolerance between two epochs (default 1e-5)" << endl
    << "-a [--abs_tol]: Absolute tolerance of loss (default 0.1)" << endl
    << "-m [--max_epoch]: Max epoch setting (default 500)" << endl
    << "-e [--estimate_n_samples]: Estimation on number of training samples."
        " Precise estimation can improve the memory usage (default 1000)" << endl
    << "-l [--learning_rate]: Learning rate setting (default 0.01)" << endl
    << "-C [--penality_base]: C base value (default 1)" << endl
    << "-c [--adjust]: <-c x1 y1 x2 y2 ...> adjust on C base value for class label 'x' with "
        "value 'y', which 'y' will be a multiplier on base value C" << endl
    << "-h [--help]: Print usage help information"
    <<endl;
}

int main(int argc, char **argv)
{
    oplin::ParamPtr param = std::make_shared<oplin::Parameter>();
    param->solver_type = 0;
    param->problem_type = 0;
    param->rela_tol = 1e-5;
    param->abs_tol = 0.1;
    param->max_epoch = 500;
    param->learning_rate = 0.01;
    param->base_C = 1;


    int bias = -1;
    size_t estimate_n_samples = 1000;
    struct option long_options[] = {
        {"solver",   required_argument, 0,  's' },
        {"problem",  required_argument, 0,  'p' },
        {"bias",     required_argument, 0,  'b' },
        {"rela_tol", required_argument, 0,  'r' },
        {"abs_tol",  required_argument, 0,  'a' },
        {"max_epoch",required_argument, 0,  'm' },
        {"learning_rate",required_argument, 0,  'l' },
        {"estimate_samples",required_argument, 0,  'e' },
        {"penality_base",required_argument, 0,  'C' },
        {"adjust",required_argument, 0,  'c' },
        {"help",     no_argument,       0,  'h' },
        {0,0,0,0}
    };

    int opt,option_index = 0;
    while ((opt = getopt_long(argc, argv, "s:p:hb:r:a:m:l:e:C:c:",
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
        case 'e':
            estimate_n_samples = atoi(optarg);
            break;
        case 'C':
            param->base_C = atof(optarg);
            break;
        case 'c':
            param->adjust_C.push_back({atof(optarg),atof(argv[optind++])});
            while(1)
            {
                if(argv[optind][0] == '-' || argc - optind <= 2 )
                    break;
                // -c x1 y1 x2 -m should not happen
                // -c x1 y1 x2 input_file model_file should not happen
                if(argc - optind == 3 || argv[optind+1][0] == '-')
                {
                    cerr << "[Error Message] use -c option with: -c x1 y1 x2 y2 ..." <<endl;
                    print_help();
                    return EXIT_FAILURE;
                }
                param->adjust_C.push_back({atof(argv[optind++]),atof(argv[optind++])});
            }
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
    if (optind + 2 != argc)
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
    oplin::DatasetPtr dataset = oplin::read_dataset(sample_file, bias, estimate_n_samples);

    // logistic regresion instance
    std::shared_ptr<oplin::LinearBase> lr= std::make_shared<oplin::LogisticRegression>();
    // train model
    clock_t start = clock();
    lr->train(dataset, param);
    std::cout << "time train:" << float(clock() -start)/CLOCKS_PER_SEC << std::endl;
    lr->export_model_to_file(model_file);


    return EXIT_SUCCESS;
}
