//
// Created by polarbabe on 18.05.19.
//

#ifndef LIBDL_MATMUL_H
#define LIBDL_MATMUL_H

#include <memory>
#include "CNode.h"
#include "Tensor.h"
#include "Utils.h"

template <typename D, int RA, int RB>
class MatMul : public CNode<D, RA + RB - 2> {

public:
    MatMul(std::shared_ptr<Tensor<D, RA>> a, std::shared_ptr<Tensor<D, RB>> b, std::weak_ptr<Tensor<D, RA + RB - 2>> t)
    : CNode<D, RA + RB - 2>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, b->gradFn}), t), a(a), b(b) {}

    static std::shared_ptr<Tensor<D, RA + RB - 2>> matmul(std::shared_ptr<Tensor<D, RA>> a,std::shared_ptr<Tensor<D, RB>> b) {
        std::array<long, RA + RB - 2> shape {};
        std::copy_n(a->eTensor.dimensions().begin(), RA - 1, shape.begin());
        std::copy_n(b->eTensor.dimensions().begin() + 1, RB - 1, shape.begin() + RA - 1);
        long size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        auto data = std::shared_ptr<D[]>(new D[size]);
        Eigen::TensorMap<Eigen::Tensor<D, RA + RB - 2>> t(data.get(), shape);
        t = a->eTensor.contract(b->eTensor, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(RA-1, 0)});
        auto result = std::make_shared<Tensor<D, RA + RB - 2>>(data, shape);
        if (a->needsGradient() || b->needsGradient())
            result->setGradFn(std::make_shared<MatMul<D, RA, RB>>(a, b, result));
        return result;
    }

    void computeGradients() override {
        if (a->gradFn.has_value()) {
            Eigen::array<Eigen::IndexPair<int>, RB - 1> d {};
            for (int i = 0; i < RB - 1; ++i)
                d[i] = Eigen::IndexPair<int>(RA + RB - 2 - i - 1, RB - i - 1);
            Eigen::Tensor<float, RA> x = CNode<D, RA + RB - 2>::grad.contract(b->eTensor, d);
            a->gradFn.value()->addGrad(x);
        }
        if (b->gradFn.has_value()) {
            Eigen::array<Eigen::IndexPair<int>, RA - 1> d {};
            for (int i = 0; i < RA - 1; ++i)
                d[i] = Eigen::IndexPair<int>(i, i);
            Eigen::Tensor<float, RB> x = a->eTensor.contract(CNode<D, RA + RB - 2>::grad, d);
            b->gradFn.value()->addGrad(x);
        }
        CNode<D, RA + RB - 2>::finishComputeGradient();
    }

private:
    std::shared_ptr<Tensor<D, RA>> a;
    std::shared_ptr<Tensor<D, RB>> b;
};

#endif //LIBDL_MATMUL_H
