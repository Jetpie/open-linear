// Logistic Regression
//
// @author: Bingqing Qu
//
// Copyright (C) 2014-2015  Bingqing Qu <sylar.qu@gmail.com>
//
// @license: See LICENSE at root directory
#ifndef LOGISTIC_H_
#define LOGISTIC_H_

#include "linear.hpp"
#include "solver.hpp"
namespace oplin{

class LogisticRegression : public LinearBase
{
public:
    LogisticRegression() : LinearBase(){};
    explicit LogisticRegression(ModelPtr model) : LinearBase(model){};
    ~LogisticRegression(void) {};
    void train(const DatasetPtr, const ParamPtr);
};

} // namespace oplin


#endif //LOGISTIC_H_
