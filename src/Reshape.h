//
// Created by superbabes on 16.06.19.
//

#ifndef LIBDL_RESHAPE_H
#define LIBDL_RESHAPE_H


#include "CNode.h"
#include "Utils.h"
#include <pybind11/stl.h>

template <typename D, int RA, int RB>
class Reshape : public CNode<D, RB> {

public:
    Reshape(const std::shared_ptr<Tensor<D, RA>> &x,
            const std::shared_ptr<Tensor<D, RB>> &result) : CNode<D, RB>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result), oldShape(x->eTensor->dimensions()), cx(x->gradFn) {}

    static std::shared_ptr<Tensor<D, RB>> reshape(const std::shared_ptr<Tensor<D, RA>> &x, std::array<long, RB> newShape) {
        for (int i = 0; i < RB; i++)
            if (newShape[i] == -1) {
                newShape[i] = -x->eTensor->size() / std::accumulate(std::begin(newShape), std::end(newShape), 1, std::multiplies<>());
                break;
            }
        auto result = std::make_shared<Tensor<D, RB>>(x->eTensor->reshape(newShape), newShape);
        if (x->needsGradient())
            result->setGradFn(std::make_shared<Reshape<D, RA, RB>>(x, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value())
            cx.value()->addGrad(CNode<D, RB>::grad->reshape(oldShape));
        CNode<D, RB>::finishComputeGradient();
    }

private:
    Eigen::array<long, RA> oldShape;
    std::optional<std::shared_ptr<CNode<D, RA>>> cx;
};


#endif //LIBDL_RESHAPE_H
