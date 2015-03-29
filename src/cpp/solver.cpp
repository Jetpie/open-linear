#include "solver.hpp"

double
L2R_LR_Solver::loss(CSR_Vector W, double *C)
{
    // loss value, initilization
    double f = W.squaredNorm() / 2.0;

    size_t n_samples = dataset_->n_samples;
    double *y = dataset_->y;

    // W^t X
    CSR_Vector WTX = w * dataset_->X;

    for(size_t i = 0; i < n_samples; ++i)
    {
        double yWTX = y[i] * wTx(i);
        if (ywTx >= 0)
            f += C[i] * log(1+exp(yWTX));
        else
            f += C[i] *(-yWTX + log(1 + exp(yWTX)));
    }
    return f;
}
