//
//  CRF++ -- Yet Another CRF toolkit
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//

#ifndef CRFPP_NODE_H_
#define CRFPP_NODE_H_

#include <vector>
#include <cmath>
#include "path.h"
#include "common.h"

#define LOG2               0.69314718055
#define MINUS_LOG_EPSILON  50

namespace CRFPP 
{

// log(exp(x) + exp(y)); this can be used recursivly
// e.g. log(exp(log(exp(x) + exp(y))) + exp(z)) = log(exp (x) + exp(y) + exp(z))
// 指数和累积到一定程度后，会超过计算机浮点值的最大值，变成 inf，这样取 log 后也是 inf。
// 为了避免这种情况，用最大值 max 去提指数和的公因子，这样就不会使某项变得过大而无法计算
// SUM = log(exp(s1) + exp(s2) + ... + exp(s100))
//     = log{exp(max) * [exp(s1-max) + exp(s2-max) + ... + exp(s100-max)]}
//     = max + log[exp(s1-max) + exp(s2-max) + ... + exp(s100-max)]
// std::log 默认是以 e 为底。
inline double logsumexp(double x, double y, bool flg) 
{
    if (flg) return y;  // init mode
    const double vmin = std::min(x, y);
    const double vmax = std::max(x, y);
    if (vmax > vmin + MINUS_LOG_EPSILON) {
        return vmax;
    } else {
        return vmax + std::log(std::exp(vmin - vmax) + 1.0);
    }
}

struct Path;

struct Node 
{
    unsigned int         x;
    unsigned short int   y;
    double               alpha;
    double               beta;
    double               cost;
    double               bestCost;
    Node                *prev;
    const int           *fvector;
    std::vector<Path *>  lpath;      // 入边集合
    std::vector<Path *>  rpath;      // 出边集合

    void calcAlpha();
    void calcBeta();
    void calcExpectation(double *expected, double, size_t) const;

    void clear() 
    {
        x = y = 0;
        alpha = beta = cost = 0.0;
        prev = 0;
        fvector = 0;
        lpath.clear();
        rpath.clear();
    }

    void shrink() 
    {
        std::vector<Path *>(lpath).swap(lpath);
        std::vector<Path *>(rpath).swap(rpath);
    }

    Node() : x(0), y(0), alpha(0.0), beta(0.0), cost(0.0), bestCost(0.0), prev(0), fvector(0) {}
};

}

#endif
