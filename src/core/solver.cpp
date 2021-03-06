#include <cmath>
#include "solver.hpp"

namespace oplin{

using std::cout;
using std::endl;
using std::cerr;
using namespace Eigen;

SolverBase::~SolverBase(){}
GradientDescent::~GradientDescent(){}


size_t
SolverBase::wolfe_line_search(ProblemPtr problem, Eigen::Ref<ColVector>w, double& alpha)
{
    return 0;
}

size_t
SolverBase::backtracking_line_search(ProblemPtr problem, Eigen::Ref<ColVector>w, double& alpha)
{
    const double c1 = 1e-4;
    double backoff = 0.5;
    const double dir_derivative = steepest_grad_.transpose() * p_;
    if (epoch_==1)
    {
        const double snorm_p = p_.squaredNorm();
        alpha = (1 / snorm_p);
        backoff = 0.1;
    }
    // check p is descent direction
    if(dir_derivative >= 0)
    {
        cerr << "SolverBase::line_search : non-descent direction is chosen in line search ("
             << __FILE__ << ", line " << __LINE__ << ")."<< endl;
        throw(std::runtime_error("non-descent direction is chosen in line search, check gradient!"));
    }

    // armijo condition solved by backtracing approach
    // f(x_k + a*p_k) <= f(x_k) + c*a*Delta(f_k)^T * p_k
    size_t iter=0;
    // cout << "++++++++++p++++++++++\n" << p_ << "\n+++++++++++++++++++++"<<endl;
    while(true)
    {
        // update w with search direction(p) and step size(alpha)
        problem->update_weights(next_w_, w, p_, alpha);
        next_loss_ = problem->loss(next_w_);
        // cout << "next_loss: " << next_loss_ << " | sufficient_desc: " << loss_+c1*dir_derivative * alpha << endl;
        if(next_loss_ <= loss_ + c1 * dir_derivative * alpha) break;
        alpha *= backoff;
        iter++;

    }

    return iter;
}

size_t
SolverBase::line_search(ProblemPtr problem, Eigen::Ref<ColVector>w, double& alpha)
{
    switch(this->line_search_choice_)
    {
        case ARMIJO_CONDITION:
        {
            return backtracking_line_search(problem,w,alpha);
        }
        case WOLFE_CONDITION:
        {
            return wolfe_line_search(problem,w,alpha);
        }
        default:
            cerr << "SolverBase::line_search : bad line search condition choice, this is program logic error ("
                 << __FILE__ << ", line " << __LINE__ << ")."<< endl;
            throw(std::runtime_error("line search bad choice!"));
            break;
    }

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
    steepest_grad_ = ColVector::Zero(w.rows(),1);
    problem->gradient(w, steepest_grad_);
    problem->regularized_gradient(w, steepest_grad_);
    // next_grad_ = ColVector::Zero(w.rows(),1);
    next_loss_ = loss_;
    next_w_ = w;
    double rela_improve = 0;
    double alpha;
    size_t iter = 0;

    line_search_choice_ = ARMIJO_CONDITION;
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

        /*** Iteration - k ***/

        /// 01 - Update line search direction p
        //     |- for steepest gradient descent method, p is simply gradient direction
        p_ = (-1) * steepest_grad_;
        //     |- line search on p and update w and loss values
        alpha = 1.0;
        iter = this->line_search(problem, w, alpha);



        /// 02 - Termination Check
        rela_improve = fabs((next_loss_ - loss_) / loss_);
        VOUT("|%5d|%15.4f|%15.6f|%5d|\n",epoch_,next_loss_,rela_improve,iter);
        if(rela_improve < param->rela_tol || next_loss_ < param->abs_tol)
        {
            // assign next_w_ to w as return value
            w.swap(next_w_);
            break;
        }

        /// 03 - Update varaiables
        problem->gradient(next_w_, steepest_grad_);
        problem->regularized_gradient(next_w_, steepest_grad_);
        loss_ = next_loss_;
        w.swap(next_w_);
        // grad_.swap(next_grad_);
    }
}

} // oplin
