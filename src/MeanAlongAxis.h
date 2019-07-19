//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_MEANALONGAXIS_H
#define LIBDL_MEANALONGAXIS_H


#include "Tensor.h"
#include "Utils.h"
#include <pybind11/stl.h>

template <typename D, int RA, int RB>
class MeanAlongAxis : public CNode<D, RA - RB> {
public:
    MeanAlongAxis(const std::optional<std::shared_ptr<CNode<D, RA>>> &a,
                  const std::shared_ptr<Tensor<D, RA - RB>> &r,
                  const std::array<int, RB> &axis,
                  const std::array<long, RA> &oldDimensions)
                  : CNode<D, RA - RB>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), axis(axis), oldDimensions(oldDimensions) {}

    static std::shared_ptr<Tensor<D, RA - RB>> mean(const std::shared_ptr<Tensor<D, RA>> &a, const std::array<int, RB> &axis) {
        std::array<long, RA - RB> newShape {};
        for (int i = 0, j = 0; i < (RA - RB); j++)
            if (notIn(j, axis))
                newShape[i++] = a->eTensor->dimension(j);
        auto result = std::make_shared<Tensor<D, RA - RB>>(a->eTensor->mean(axis), newShape);
        if (a->needsGradient())
            result->setGradFn(std::make_shared<MeanAlongAxis<D, RA, RB>>(a->gradFn, result, axis, static_cast<std::array<long, RA>>(a->eTensor->dimensions())));
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
            int scale = std::accumulate(std::begin(broadcast), std::end(broadcast), 1, std::multiplies<>());
            auto t = CNode<D, RA - RB>::grad->reshape(reshape).broadcast(broadcast);
            a.value()->addGrad(t / t.constant(scale));
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


#endif //LIBDL_MEANALONGAXIS_H


/*//
// Created by polarbabe on 22.05.19.
//

#ifndef LIBDL_MEANALONGAXIS_H
#define LIBDL_MEANALONGAXIS_H


#include "Tensor.h"
#include "Utils.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

template <typename D, int RA, int RB>
class MeanAlongAxis : public CNode<D, RA - RB> {
public:
    MeanAlongAxis(const std::optional<std::shared_ptr<CNode<D, RA>>> &a,
                  const std::shared_ptr<Tensor<D, RA - RB>> &r,
                  int axis,
                  long size)
                  : CNode<D, RA - RB>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a}), r), a(a), axis(axis), size(size) {}

    static std::shared_ptr<Tensor<D, RA - RB>> mean(const std::shared_ptr<Tensor<D, RA>> &a, const std::array<int, RB> &axis) {
        auto result = std::make_shared<Tensor<D, RA - RB>>(a->eTensor->mean(axis));
        if (a->needsGradient())
            result->setGradFn(std::make_shared<MeanAlongAxis<D, RA, RB>>(a->gradFn, result, axis, a->eTensor->dimension(axis)));
        return result;
    }

    void computeGradients() override {
        if (a.has_value()) {
            std::array<int, R + 1> reshape;
            std::array<int, R + 1> broadcast;
            for (int i = 0; i < axis; ++i) {
                reshape[i] = CNode<D, R>::grad->dimension(i);
                broadcast[i] = 1;
            }
            reshape[axis] = 1;
            broadcast[axis] = size;
            for (int i = axis + 1; i < R + 1; ++i) {
                reshape[i] = CNode<D, R>::grad->dimension(i - 1);
                broadcast[i] = 1;
            }
            auto t = CNode<D, R>::grad->reshape(reshape).broadcast(broadcast);
            a.value()->addGrad(t / t.constant(size));
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R + 1>>> a;
    int axis;
    long size;
};

#endif //LIBDL_MEANALONGAXIS_H*/