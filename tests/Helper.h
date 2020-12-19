
#ifndef LIBDL_HELPER_H
#define LIBDL_HELPER_H

#include <iostream>
#include "../src/Tensor.h"

using namespace Catch::literals;

template<std::int64_t N>
auto makeTensor(const std::int64_t (&dimensions)[N], bool requiresGrad = true) {
    std::array<std::int64_t, N> d;
    std::copy_n(std::begin(dimensions), N, std::begin(d));
    auto ret = std::make_shared<Tensor<std::float_t, N>>(d, requiresGrad);
    if (requiresGrad)
        ret->setGradFn(std::make_shared<Leaf<std::float_t, N>>(ret));
    return ret;
}

template<std::int64_t N>
auto trange(const std::int64_t (&dimensions)[N], bool requiresGrad = true, std::int64_t mod = 1000) {
    auto ret = makeTensor(dimensions, requiresGrad);
    for (std::int64_t i = 0; i < ret->data->size(); i++)
        ret->data->data()[i] = static_cast<std::float_t>((i % mod) + 1);
    return ret;
}

template<std::int64_t N>
auto constant(const std::int64_t (&dimensions)[N], std::float_t value, bool requiresGrad = true) {
    auto ret = makeTensor(dimensions, requiresGrad);
    ret->data->setConstant(value);
    return ret;
}

template<std::int64_t N>
auto random(const std::int64_t (&dimensions)[N], bool requiresGrad = true) {
    auto ret = makeTensor(dimensions, requiresGrad);
    *ret->data = ret->data->random() * ret->data->constant(2.) - ret->data->constant(1.);
    return ret;
}

template<std::int64_t N>
auto setGradAndBackward(const std::shared_ptr<Tensor<std::float_t, N>> &t, std::int64_t mod = 1000) {
    Eigen::Tensor<std::float_t, N> grad(t->data->dimensions());
    for (std::int64_t i = 0; i < grad.size(); i++)
        grad.data()[i] = static_cast<std::float_t>((i % mod) + 1);
    t->gradFn.value()->addGrad(grad);
    t->backward(0);
    return grad;
}

template<std::int64_t R>
bool tensorEqual(const Eigen::Tensor<std::float_t, R> &a, const Eigen::Tensor<std::float_t, R> &b, std::float_t atol = 1e-6) {

    if (a.size() != b.size())
        return false;

    for (std::int64_t i = 0; i < R; i++)
        if (a.dimension(i) != b.dimension(i))
            return false;

    for (std::int64_t i = 0; i < a.size(); i++)
        if (abs(a.data()[i] - b.data()[i]) > atol)
            return false;

    return true;
}

#endif //LIBDL_HELPER_H
