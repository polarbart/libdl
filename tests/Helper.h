
#ifndef LIBDL_HELPER_H
#define LIBDL_HELPER_H

#include <iostream>
#include "../src/Tensor.h"

using namespace Catch::literals;

template<long N>
auto makeTensor(const long (&dimensions)[N], bool requiresGrad = true) {
    std::array<long, N> d;
    std::copy_n(std::begin(dimensions), N, std::begin(d));
    auto ret = std::make_shared<Tensor<float, N>>(d, requiresGrad);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<float, N>>(ret));
    return ret;
}

template<long N>
auto trange(const long (&dimensions)[N], bool requiresGrad = true, int mod = 1000) {
    auto ret = makeTensor(dimensions, requiresGrad);
    for (int i = 0; i < ret->data->size(); i++)
        ret->data->data()[i] = (i % mod) + 1;
    return ret;
}

template<long N>
auto constant(const long (&dimensions)[N], float value, bool requiresGrad = true) {
    auto ret = makeTensor(dimensions, requiresGrad);
    ret->data->setConstant(value);
    return ret;
}

template<long N>
auto random(const long (&dimensions)[N], bool requiresGrad = true) {
    auto ret = makeTensor(dimensions, requiresGrad);
    *ret->data = ret->data->random() * ret->data->constant(2) - ret->data->constant(1);
    return ret;
}

template<long N>
auto setGradAndBackward(const std::shared_ptr<Tensor<float, N>> &t, int mod = 1000) {
    Eigen::Tensor<float, N> grad(t->data->dimensions());
    for (int i = 0; i < grad.size(); i++)
        grad.data()[i] = (i % mod) + 1;
    t->gradFn.value()->addGrad(grad);
    t->backward(0);
    return grad;
}

template<int R>
bool tensorEqual(const Eigen::Tensor<float, R> &a, const Eigen::Tensor<float, R> &b, float atol = 1e-6) {

    if (a.size() != b.size())
        return false;

    for (int i = 0; i < R; i++)
        if (a.dimension(i) != b.dimension(i))
            return false;

    for (int i = 0; i < a.size(); i++)
        if (abs(a.data()[i] - b.data()[i]) > atol)
            return false;

    return true;
}

#endif //LIBDL_HELPER_H
