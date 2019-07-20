//
// Created by polarbabe on 18.05.19.
//

#ifndef LIBDL_MATMUL_H
#define LIBDL_MATMUL_H

#include <memory>
#include "CNode.h"
#include "../Tensor.h"
#include "../Utils.h"

template <typename D, int RA, int RB>
class MatMul : public CNode<D, RA + RB - 2> {

public:
    MatMul(const std::shared_ptr<Tensor<D, RA>> &a,
           const std::shared_ptr<Tensor<D, RB>> &b,
           const std::shared_ptr<Tensor<D, RA + RB - 2>> &t)
    : CNode<D, RA + RB - 2>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, b->gradFn}), t), a(a->eTensor), b(b->eTensor), ca(a->gradFn), cb(b->gradFn) {}

    static std::shared_ptr<Tensor<D, RA + RB - 2>> matmul(const std::shared_ptr<Tensor<D, RA>> &a, const std::shared_ptr<Tensor<D, RB>> &b) {
        std::array<long, RA + RB - 2> shape {};
        std::copy_n(std::begin(a->eTensor->dimensions()), RA - 1, std::begin(shape));
        std::copy_n(std::begin(b->eTensor->dimensions()) + 1, RB - 1, std::begin(shape) + RA - 1);

        auto t = a->eTensor->contract(*b->eTensor, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(RA-1, 0)});
        auto result = std::make_shared<Tensor<D, RA + RB - 2>>(t, shape);

        if (a->needsGradient() || b->needsGradient())
            result->setGradFn(std::make_shared<MatMul<D, RA, RB>>(a, b, result));
        return result;
    }

    void computeGradients() override {
        if (ca.has_value()) {
            Eigen::array<Eigen::IndexPair<int>, RB - 1> d {};
            for (int i = 0; i < RB - 1; ++i)
                d[i] = Eigen::IndexPair<int>(RA + RB - 2 - i - 1, RB - i - 1);
            ca.value()->addGrad(CNode<D, RA + RB - 2>::grad->contract(*b, d));
        }
        if (cb.has_value()) {
            Eigen::array<Eigen::IndexPair<int>, RA - 1> d {};
            for (int i = 0; i < RA - 1; ++i)
                d[i] = Eigen::IndexPair<int>(i, i);
            cb.value()->addGrad(a->contract(*CNode<D, RA + RB - 2>::grad, d));
        }
        CNode<D, RA + RB - 2>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, RA>>> a;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, RB>>> b;
    std::optional<std::shared_ptr<CNode<D, RA>>> ca;
    std::optional<std::shared_ptr<CNode<D, RB>>> cb;
};

#endif //LIBDL_MATMUL_H
