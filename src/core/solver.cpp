#include <cmath>
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
 * @param problem Problem instance
 * @param w       weights for optimization
 * @param alpha   step size of line search
 *
 */
size_t
SolverBase::line_search(ProblemPtr problem, Eigen::Ref<ColVector>w, double& alpha)
{
    const double c1 = 1e-4;
    const double backoff = 0.5;
    const size_t max_iter = 10;
    const double dir_dirivative = grad_.transpose() * p_;

    // check p is descent direction
    if(dir_dirivative >= 0)
    {
        VOUT("\n");
        cerr << "SolverBase::line_search : non-descent direction is chosen in line search ("
             << __FILE__ << ", line " << __LINE__ << ")."<< endl;
        throw(std::runtime_error("non-descent direction is chosen in line search, check gradient!"));
    }

    // precompute
    const double sufficient_desc = prev_loss_ + dir_dirivative * c1 * alpha;

    // armijo condition solved by backtracing approach
    // f(x_k + a*p_k) <= f(x_k) + c*a*Delta(f_k)^T * p_k
    size_t iter=0;
    for (; iter < max_iter; ++iter)
    {
        // update w with search direction(p) and step size(alpha)
        problem->update_weights(w, prev_w_, p_, alpha);
        loss_ = problem->loss(w);
        if(loss_ <= sufficient_desc) break;
        alpha *= backoff;
    }

    return iter;
}


/**
 * Solve the problem on dataset with parameters
 *
 * @param problem Problem instance
 * @param param   Parameter instance
 * @param w       weights for optimize
 *
 */
void
GradientDescent::solve(ProblemPtr problem, ParamPtr param, Eigen::Ref<ColVector>& w)
{

    // initializations
    loss_ = problem->loss(w);
    grad_ = ColVector::Zero(problem->dataset_->dimension,1);
    prev_w_ = w;
    prev_loss_ = loss_;
    double rela_improve = 0;
    double alpha;
    size_t iter = 0;

    // check if the weights already optimized
    if(loss_ < param->abs_tol)
    {
        return;
        VOUT("Already optimized weights get!");
    }

    // -- debug print
    VOUT("\n*** To disable debug info: $ make DISABLE_DEBUG=yes ***\n");
    VOUT("|%5s|%15s|%15s|%5s|\n","Epoch","Loss","Improve","#iter");

    for(epoch_ = 0; epoch_ < param->max_epoch; ++epoch_)
    {
        /// 01 - Update gradient and line search direction p
        //     |- for steepest gradient descent method, p is simply gradient direction
        problem->gradient(prev_w_, grad_);
        p_ = (-1) * grad_;
        alpha = 1.0;
        //     |- line search on p and update w and loss values
        iter = this->line_search(problem, w, alpha);


        /// 02 - Termination Check
        rela_improve = fabs(loss_ - prev_loss_);
        VOUT("|%5d|%15.4f|%15.6f|%5d|\n",epoch_,loss_,rela_improve,iter);
        if(rela_improve < param->rela_tol || loss_ < param->abs_tol) break;


        /// 03 - Update varaiables
        prev_loss_ = loss_;
        w.swap(prev_w_);
    }
}

} // oplin
