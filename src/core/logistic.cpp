// Logistic Regression
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#include "logistic.hpp"

namespace oplin{
using std::cout;
using std::endl;
using std::cerr;
using namespace Eigen;
/**
 * Train the coefficients using depends on the parameters.
 *
 * @param dataset training dataset
 * @param param parameters
 *
 */
void
LogisticRegression::train(const DatasetPtr dataset, const ParamPtr param)
{
    // input sanity check
    // TODO : validation function on dataset
    if(!param)
    {
        cerr << "LogisticRegression::train : Error input, param NULL or not valid, "
             << __FILE__ << "," << __LINE__ << endl;
        throw(std::invalid_argument("param not valid"));
    }
    if(!dataset)
    {
        cerr << "LogisticRegression::train : Error input, dataset NULL or not valid, "
             << __FILE__ << "," << __LINE__ << endl;
        throw(std::invalid_argument("dataset not valid"));
    }

    size_t n_samples = dataset->n_samples;
    size_t dimension = dataset->dimension;
    size_t n_classes = dataset->n_classes;

    // initialize model
    ModelUniPtr model(new Model());
    // sanity check
    if(!model)
    {
        cerr << "LogisticRegression::train : Model initialization error (memory), "
             << __FILE__ << "," << __LINE__ << endl;
        throw(std::bad_alloc());
    }

    model->dimension = dimension;
    model->n_classes = n_classes;

    std::vector<size_t> count;
    std::vector<size_t> start_idx;
    std::vector<size_t> perm_idx;
    count.reserve(n_classes);
    start_idx.reserve(n_classes);
    perm_idx.reserve(n_samples);

    // adjust the label sequence to if n_classes = 2 to keep +1 label
    // in advance
    if(n_classes==2 && dataset->labels[0] == -1 && dataset->labels[1] == 1)
        std::swap(dataset->labels[0],dataset->labels[1]);

    model->labels = dataset->labels;

    // preprocess the training dataset
    try
    {
        preprocess_data(dataset, count, start_idx, perm_idx);
    }
    catch(std::out_of_range& e)
    {
        throw(std::runtime_error("Preprocessing(grouping) dataset fail!"));
    }

    // TODO : test the actual memory usage of following manipulations
    // permute the instances
    // -construct the Eigen permutation matrix
    PermutationMatrix<Dynamic,Dynamic> perm_matrix(n_samples);
    for(size_t i = 0; i < n_samples ;++i)
        perm_matrix.indices()[i] = perm_idx[i];
    // -permutation columns
    *(dataset->X) = (*(dataset->X) * perm_matrix).eval();

    size_t k;
    // construct multiplier for different classes
    std::vector<double> penality_weights(n_classes, param->base_C);
    for(std::vector<KeyValue<double,double> >::iterator it = param->adjust_C.begin();
        it!=param->adjust_C.end(); ++it)
    {
        double label = (*it).i;
        for(k=0;k < n_classes;++k)
        {
            if(label == dataset->labels[k])
                break;
        }
        if(k == n_classes)
        {
            cout << "Warning : No matched label ["<<label<<"] for adjust_C(ignored), "
                 << __FILE__ << "," << __LINE__ << endl;
            continue;
        }
        penality_weights[k] *= (*it).v;
    }

    size_t n_ws = n_classes == 2? 1: n_classes;
    double* W_ = new double[dimension * n_ws];
    // sanity check
    if(!W_)
    {
        cerr << "LogisticRegression::train : W_ initialization error (memory), "
             << __FILE__ << "," << __LINE__ << endl;
        throw(std::bad_alloc());
    }
    // handle two class classification problem
    if(n_classes == 2)
    {
        // initialize weights to -0.5 ~ 0.5
        // srand((unsigned int) time(0));
        // ColVector w = ColVector::Random(dimension,1) / 2;
        ColVector w = ColVector::Zero(dimension,1);

        std::vector<double> C(n_samples, penality_weights[1]);
        // rearrange labels
        size_t pos_end = start_idx[0] + count[0];
        for(k=0;k<pos_end;++k)
        {
            dataset->y[k] = +1;

            // adjust the penality value for positive samples (rare)
            C[k] = penality_weights[0];
        }
        for(;k<n_samples;++k)
        {
            dataset->y[k] = -1;
        }

        train_ovr(dataset, param, C, w);
        for(size_t idx = 0; idx < dimension ;++idx)
            W_[idx] = w(idx);
        VOUT("#non-zeros / #features : %d / %d\n",(w.array() != 0).count(), dimension);
    }
    // multiple class using one-vs-rest strategy
    else
    {
        // TODO: implementation
        cout << ">2 classes" << endl;
    }

    // load parameter pointer into model, this is good practice if
    // this is production environment, in which read and write
    // manipulations are quite sensitive.
    model->set_weights(W_);
    model->bias = dataset->bias;
    if(model->bias > 0)
    {
        double* bias_values = new double[n_ws];
        if(!bias_values)
        {
            cerr << "LogisticRegression::train : bias initialization error (memory), "
                 << __FILE__ << "," << __LINE__ << endl;
            delete [] W_;
            throw(std::bad_alloc());
        }
        // store w_0 * bias term
        for(size_t i = 0; i < n_ws;++i)
        {
            bias_values[i] = dataset->bias * W_[dimension*(i+1)-1];
        }
        model->set_bias_values(bias_values);
    }
    // Finally, pass model variable to member model
    this->load_model( std::move(model) );
}

/**
 * Train One-vs-Rest
 *
 *
 */
void
LogisticRegression::train_ovr(DatasetPtr dataset, ParamPtr param,const std::vector<double>& C, Eigen::Ref<ColVector> w)
{
    std::shared_ptr<Problem> problem;
    std::shared_ptr<SolverBase> solver;
    // make problem
    switch(param->problem_type)
    {
        case L1R_LR:
        {
            problem = std::make_shared<L1R_LR_Problem>(dataset,C);
            break;
        }
        case L2R_LR:
        {
            problem = std::make_shared<L2R_LR_Problem>(dataset,C);
            break;
        }
        default:
            cerr << "LogisticRegression::train_ovr : invalid problem type, "
                 << "Default option (L2R_LR) will be used, "
                 << __FILE__ << "," << __LINE__ << endl;
            problem = std::make_shared<L2R_LR_Problem>(dataset,C);
            break;
    }

    // decide solver
    switch(param->solver_type)
    {
        case GD:
        {
            solver = std::make_shared<GradientDescent>();
            break;
        }
        case L_BFGS:
        {
            solver = std::make_shared<oplin::LBFGS>();
            break;
        }
        default:
            cerr << "LogisticRegression::train : invalid solver type, "
                 << "Default option (LBFGS) will be used, "
                 << __FILE__ << "," << __LINE__ << endl;
            // TODO : add default initialization on solver
            break;
    }

    solver->solve(problem, param, w);

    cout << "--------------Dev Print--------------" << endl;
    if(w.rows()<100)
    {
        cout <<"weights:"<<endl;
        cout << w <<endl;
    }
    cout << "-----------------End-----------------" << endl;
    return;
}

} // oplin
