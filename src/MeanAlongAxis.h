//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_MEANALONGAXIS_H
#define LIBDL_MEANALONGAXIS_H


#include "Tensor.h"
#include "Utils.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename D, int R>
class MeanAlongAxis : public CNode<D, R> {
public:
    MeanAlongAxis(std::optional<std::shared_ptr<CNode<D, R + 1>>> a, std::shared_ptr<Tensor<D, R>> r, int axis, long size)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), axis(axis), size(size) {}

    static std::shared_ptr<Tensor<D, R - 1>> mean(std::shared_ptr<Tensor<D, R>> a, int axis) {
        std::array<long, R - 1> shape {};
        std::copy_n(a->eTensor.dimensions().begin(), axis, shape.begin());
        std::copy_n(a->eTensor.dimensions().begin() + axis + 1, R - axis - 1, shape.begin() + axis);
        auto data = std::shared_ptr<D[]>(new D[a->eTensor.size() / a->eTensor.dimension(axis)]);
        Eigen::TensorMap<Eigen::Tensor<D, R - 1>> t(data.get(), shape);
        t = a->eTensor.mean(Eigen::array<int, 1>{axis});
        auto result = std::make_shared<Tensor<D, R - 1>>(data, shape);
        if (a->needsGradient())
            result->setGradFn(std::make_shared<MeanAlongAxis<D, R - 1>>(a->gradFn, result, axis, a->eTensor.dimension(axis)));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            std::array<int, R + 1> reshape;
            std::array<int, R + 1> broadcast;
            for (int i = 0; i < axis; ++i) {
                reshape[i] = CNode<D, R>::grad.dimension(i);
                broadcast[i] = 1;
            }
            reshape[axis] = 1;
            broadcast[axis] = size;
            for (int i = axis + 1; i < R + 1; ++i) {
                reshape[i] = CNode<D, R>::grad.dimension(i - 1);
                broadcast[i] = 1;
            }
            auto t = CNode<D, R>::grad.reshape(reshape).broadcast(broadcast);
            a.value()->addGrad(t / t.constant(size));
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R + 1>>> a;
    int axis;
    long size;
};


#endif //LIBDL_MEANALONGAXIS_H
