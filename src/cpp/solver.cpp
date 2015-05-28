#include "solver.hpp"

namespace oplin{

using std::cout;
using std::endl;
using std::cerr;
using namespace Eigen;
SolverBase::~SolverBase(){}
GradientDescent::~GradientDescent(){}

/**
 * Line search
 *
 *
 *
 */
void
SolverBase::line_search(const ProblemPtr problem, const ParamPtr param,
                        const ColVector& p, double& alpha)
{

}


/**
 * Solve the problem on dataset with parameters
 * @param problem
 * @param param
 * @param w
 */
void
GradientDescent::solve(ProblemPtr problem, ParamPtr param, ColVector& w)
{
    double learning_rate = param->learning_rate;
    size_t epoch = 0;
    double last_loss = 0;
    double rela_improve = 0;

    while(epoch < param->max_epoch)
    {
        double loss = problem->loss(w, param->C);
        rela_improve = ABS(loss - last_loss);

        cout << "current loss for epoch " << epoch << " : " << loss << endl;
        cout << "relative improvement for epoch " << epoch << " : "
             << rela_improve << endl;

        if(rela_improve < param->rela_tol || loss < param->abs_tol)
            break;
        //
        last_loss = loss;
        //
        w = w - learning_rate * problem->gradient(w, param->C);

        ++epoch;
    }
}

} // oplin
