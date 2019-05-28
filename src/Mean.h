//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_MEAN_H
#define LIBDL_MEAN_H


#include "Tensor.h"
#include "Utils.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename D, int K>
class Mean : public CNode<D, 0> {
public:
    Mean(const std::optional<std::shared_ptr<CNode<D, K>>> &a,
         const std::shared_ptr<Tensor<D, 0>> &r, const std::array<long, K> &shape)
            : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), shape(shape) {}

    static std::shared_ptr<Tensor<D, 0>> mean(const std::shared_ptr<Tensor<D, K>> &a) {
        auto result = std::make_shared<Tensor<D, 0>>(a->eTensor->mean(), std::array<long, 0> {});
        if (a->needsGradient())
            result->setGradFn(std::make_shared<Mean<D, K>>(a->gradFn, result, a->eTensor->dimensions()));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            std::array<long, K> r;
            r.fill(1);
            auto t = CNode<D, 0>::grad->reshape(r).broadcast(shape);
            long size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
            a.value()->addGrad(t / t.constant(size));
        }
        CNode<D, 0>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, K>>> a;
    std::array<long, K> shape;
};


#endif //LIBDL_MEAN_H
