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


class LogisticRegression : protected LinearBase
{
public:
    ~LogisticRegression(void) {};

    void train(const DatasetPtr, const ParamPtr);
    void predict_proba();

};

#endif //LOGISTIC_H_
