
#ifndef LIBDL_TEST_HELPER_H
#define LIBDL_TEST_HELPER_H

#include <iostream>
#include <catch2/catch.hpp>
#include "../src/ops/Add.h"


template<std::size_t N>
auto make_tensor(const long (&dimensions)[N], bool requiresGrad = true) {
    std::array<long, N> d;
    std::copy_n(std::begin(dimensions), N, std::begin(d));
    auto ret = std::make_shared<Tensor<float, N>>(d, requiresGrad);
    for (int i = 0; i < ret->eTensor->size(); i++)
        ret->eTensor->data()[i] = i+1;
    return ret;
}

#endif //LIBDL_TEST_HELPER_H
