//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_SUMALONGAXIS_H
#define LIBDL_SUMALONGAXIS_H


#include "../Tensor.h"
#include "../Utils.h"
#include <pybind11/stl.h>

template <typename D, int RA, int RB>
class SumAlongAxis : public CNode<D, RA - RB> {
public:
    SumAlongAxis(const std::optional<std::shared_ptr<CNode<D, RA>>> &a,
                  const std::shared_ptr<Tensor<D, RA - RB>> &r,
                  const std::array<int, RB> &axis,
                  const std::array<long, RA> &oldDimensions)
            : CNode<D, RA - RB>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), axis(axis), oldDimensions(oldDimensions) {}

    static std::shared_ptr<Tensor<D, RA - RB>> mean(const std::shared_ptr<Tensor<D, RA>> &a, const std::array<int, RB> &axis) {
        std::array<long, RA - RB> newShape {};
        for (int i = 0, j = 0; i < (RA - RB); j++)
            if (notIn(j, axis))
                newShape[i++] = a->eTensor->dimension(j);
        auto result = std::make_shared<Tensor<D, RA - RB>>(a->eTensor->sum(axis), newShape);
        if (a->needsGradient())
            result->setGradFn(std::make_shared<SumAlongAxis<D, RA, RB>>(a->gradFn, result, axis, static_cast<std::array<long, RA>>(a->eTensor->dimensions())));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            std::array<int, RA> reshape;
            std::array<int, RA> broadcast;
            for (int i = 0; i < RA; i++) {
                if (notIn(i, axis)) {
                    reshape[i] = oldDimensions[i];
                    broadcast[i] = 1;
                } else {
                    reshape[i] = 1;
                    broadcast[i] = oldDimensions[i];
                }
            }
            a.value()->addGrad(CNode<D, RA - RB>::grad->reshape(reshape).broadcast(broadcast));
        }
        CNode<D, RA - RB>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, RA>>> a;
    std::array<int, RB> axis;
    std::array<long, RA> oldDimensions;

    static bool notIn(int a, const std::array<int, RB> &axis) {
        for (auto i : axis)
            if (a == i)
                return false;
        return true;
    }
};

#endif //LIBDL_SUMALONGAXIS_H
