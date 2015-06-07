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
                        const Eigen::Ref<ColVector>& p, double& alpha)
{

}


/**
 * Solve the problem on dataset with parameters
 * @param problem
 * @param param
 * @param w
 */
void
GradientDescent::solve(ProblemPtr problem, ParamPtr param, Eigen::Ref<ColVector>& w, std::vector<double>& C)
{
    double learning_rate = param->learning_rate;
    size_t epoch = 0;
    double last_loss = 0;
    double rela_improve = 0;

    while(epoch < param->max_epoch)
    {
        double loss = problem->loss(w, C);
        rela_improve = oplin::ABS(loss - last_loss);

        VOUT("Epoch(%d) - loss : %f / relative improvement : %f\n" ,
             epoch,loss,rela_improve);

        // stop criteria
        if(rela_improve < param->rela_tol || loss < param->abs_tol)
            break;
        //
        last_loss = loss;
        //
        w.noalias() -= learning_rate * problem->gradient(w, C);

        ++epoch;
    }
}

} // oplin
