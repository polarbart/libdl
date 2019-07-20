//
// Created by superbabes on 06.07.19.
//

#ifndef LIBDL_LINEAR_H
#define LIBDL_LINEAR_H


#include <memory>
#include "CNode.h"
#include "../Tensor.h"
#include "../Utils.h"

#define R 2

template <typename D>
class Linear : public CNode<D, R> {

public:
    Linear(const std::shared_ptr<Tensor<D, R>> &w,
           const std::shared_ptr<Tensor<D, R>> &x,
           const std::shared_ptr<Tensor<D, 1>> &b,
           const std::shared_ptr<Tensor<D, R>> &result)
           : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({w->gradFn, x->gradFn, b->gradFn}), result),
           w(w->eTensor),
           x(x->eTensor),
           cw(w->gradFn),
           cx(x->gradFn),
           cb(b->gradFn) {}

    Linear(const std::shared_ptr<Tensor<D, R>> &w,
           const std::shared_ptr<Tensor<D, R>> &x,
           const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({w->gradFn, x->gradFn}), result),
              w(w->eTensor),
              x(x->eTensor),
              cw(w->gradFn),
              cx(x->gradFn),
              cb(std::nullopt) {}

   /*
    * w (f, f')
    * x (f, n)
    * b (f')
    * result (f', n)
    * */
    static std::shared_ptr<Tensor<D, R>> linear(const std::shared_ptr<Tensor<D, R>> &w, const std::shared_ptr<Tensor<D, R>> &x, const std::shared_ptr<Tensor<D, 1>> &b) {
        std::array<long, R> shape {w->eTensor->dimension(1), x->eTensor->dimension(1)};

        auto t = w->eTensor->contract(*x->eTensor, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});
        std::shared_ptr<Tensor<D, R>> result;

        if (b != nullptr) {
            Eigen::array<long, R> reshape {shape[0], 1};
            Eigen::array<long, R> broadcast {1, shape[1]};
            result = std::make_shared<Tensor<D, R>>(t + b->eTensor->reshape(reshape).broadcast(broadcast), shape);
        } else
            result = std::make_shared<Tensor<D, R>>(t, shape);

        if (w->needsGradient() || x->needsGradient() || (b != nullptr && b->needsGradient())) {
            if (b != nullptr)
                result->setGradFn(std::make_shared<Linear<D>>(w, x, b, result));
            else
                result->setGradFn(std::make_shared<Linear<D>>(w, x, result));
        }
        return result;
    }

    void computeGradients() override {
        if (cw.has_value())
            cw.value()->addGrad(x->contract(*CNode<D, R>::grad, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 1)}));
        if (cx.has_value())
            cx.value()->addGrad(w->contract(*CNode<D, R>::grad, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(1, 0)}));
        if (cb.has_value())
            cb.value()->addGrad(CNode<D, R>::grad->sum(Eigen::array<int, 1> {1}));
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> w;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> x;
    std::optional<std::shared_ptr<CNode<D, R>>> cw;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::optional<std::shared_ptr<CNode<D, 1>>> cb;
};

#undef R

#endif //LIBDL_LINEAR_H
