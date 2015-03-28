// Solver formulations
//
// Naming Convention: solver classes are named by all capital letter denote
//                    type and a "Solver" prefix separated by underscore
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory

#include <string>
#include <vector>

#include <Eigen/Dense>
using namespace std;
using namespace Eigen

typedef Eigen::SparseMatrix<double> CSR_Matrix;
typedef Eigen::SparseVector<vector> CSR_Vector;

///
///
class SolverBase
{
public:
    virtual ~function(void) {}

    virtual double loss() = 0;
    virtual void gradient() = 0;

};

///
///
class L2R_LR_Solver : public SolverBase
{
private:
    const DatasetPtr
public:
    L2R_LR_Solver(const DatasetPtr);
    ~L2R_LR_Solver();

    double loss(CSR_Matrix);
    CSR_Matrix gradient(CSR_Matrix);
};
