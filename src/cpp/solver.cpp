#include "solver.hpp"

double
L2R_LR_Solver::loss(CSR_Vector w, double *C)
{
    // loss value, initilization
    double f = w.squaredNorm() / 2.0;

    size_t n_samples = dataset_->n_samples;
    int *y = dataset_->y;

    // W^t X
    CSR_Vector wTX = w * dataset_->X;

    for(size_t i = 0; i < n_samples; ++i)
    {
        double ywTX = y[i] * wTX.coeffRef(i);
        if (ywTX >= 0)
            f += C[i] * log(1+exp(ywTX));
        else
            f += C[i] *(-ywTX + log(1 + exp(ywTX)));
    }
    return f;
}
