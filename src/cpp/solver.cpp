#include "solver.hpp"

SolverBase::~SolverBase(){}
GradientDescent::~GradientDescent(){}

void
GradientDescent::solve(ProblemPtr problem, ParamPtr param, ColVector& w)
{
    double lambda = 0.01;
    size_t epoch = 0;
    double last_loss = 0;
    double rela_improve = 0;
    while(epoch < param->max_epoch)
    {
        double loss = problem->loss(w, param->C);

        rela_improve = loss - last_loss;
        rela_improve = rela_improve < 0? -rela_improve : rela_improve;

        cout << "current loss for epoch " << epoch << " : " << loss << endl;
        cout << "relative improvement for epoch " << epoch << " : " << rela_improve << endl;

        if(rela_improve < param->rela_tol || loss < param->abs_tol)
            break;
        //
        last_loss = loss;
        //
        ColVector tri_w = problem->gradient(w, param->C);
        //
        w = w - lambda * tri_w;

        ++epoch;
    }
}
