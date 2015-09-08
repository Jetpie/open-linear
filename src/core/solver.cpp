#include "solver.hpp"

namespace oplin{

using std::cout;
using std::endl;
using std::cerr;
using namespace Eigen;

SolverBase::~SolverBase(){}
GradientDescent::~GradientDescent(){}
/**
 * Line search:
 *   The basic line search form would be: x_{k+1} = x_k + p_k * alpha_k
 *   The typical form of p will be p_k = -B_k^(-1) * grad(f_k)
 *   B_k is always a symmetric and nonsingular matrix.
 *
 * @param problem
 * @param param
 * @param p
 * @param alpha
 */
double
SolverBase::line_search(ProblemPtr problem, const Eigen::Ref<ColVector>& w, const Eigen::Ref<ColVector>& grad,
                        const Eigen::Ref<ColVector>& p, const double loss)
{
    double alpha = 1.0;
    const double c1 = 1e-4;
    const double ro = 0.5;
    size_t epoch = 0;
    const size_t max_epoch = 10;
    const double dir_dirivative = grad.transpose() * p;
    // check p is descent direction
    if(dir_dirivative >= 0)
    {
        VOUT("\n");
        cerr << "SolverBase::line_search : non-descent direction is chosen in line search ("
             << __FILE__ << ", line " << __LINE__ << ")."<< endl;
        throw(std::runtime_error("non-descent direction is chosen in line search, check gradient!"));
    }
    // precompute
    const double sufficient_desc = loss + dir_dirivative * c1 * alpha;
    // armijo condition solved by backtracing approach
    // f(x_k + a*p_k) <= f(x_k) + c*a*Delta(f_k)^T * p_k
    while(epoch++ < max_epoch)
    {
        if(problem->loss(w + alpha * p) <= sufficient_desc)
            break;
        alpha *= ro;
    }
    VOUT("#iter(line search): %d\n",epoch);
    return alpha;
}


/**
 * Solve the problem on dataset with parameters
 *
 * @param problem
 * @param param
 * @param w
 */
void
GradientDescent::solve(ProblemPtr problem, ParamPtr param, Eigen::Ref<ColVector>& w)
{
    size_t epoch = 0;
    double last_loss = 0;
    double rela_improve = 0;
    double alpha;
    while(epoch++ < param->max_epoch)
    {
        double loss = problem->loss(w);
        rela_improve = oplin::ABS(loss - last_loss);

        VOUT("Epoch(%d) - loss : %f | relative improvement : %f | " ,
             epoch,loss,rela_improve);

        // stop criteria
        if(rela_improve < param->rela_tol || loss < param->abs_tol)
        {
            VOUT("\n");
            break;
        }
        //
        last_loss = loss;
        // for steepest gradient descent method, p is simply gradient direction
        ColVector grad = problem->gradient(w);
        ColVector p = (-1) * grad;
        alpha = this->line_search(problem, w, grad, p, loss);
        w.noalias() += alpha * p;
    }
}

} // oplin
