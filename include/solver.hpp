// Solver formulations
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

class SolverBase
{
public:
    virtual ~function(void) {}

    virtual double loss() = 0;
    virtual void gradient() = 0;

};
