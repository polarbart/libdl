
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
    Linear(
            const std::shared_ptr<Tensor<D, R>> &w,
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &b,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({w->gradFn, x->gradFn, b->gradFn}), result),
            w(w->data),
            x(x->data),
            cw(w->gradFn),
            cx(x->gradFn),
            cb(b->gradFn) {}

    Linear(
            const std::shared_ptr<Tensor<D, R>> &w,
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({w->gradFn, x->gradFn}), result),
            w(w->data),
            x(x->data),
            cw(w->gradFn),
            cx(x->gradFn),
            cb(std::nullopt) {}

   /*
    * \brief linear transformation w^T*x + b
    *
    * \param w a 2d weight tensor with shape (f, f')
    * \param x the 2d tensor which should be transformed linearly of shape (f, batchsize)
    * \param b a 1d bias tensor with shape (f',), may be null
    *
    * \return a new tensor of shape (f', batchsize)
    * */
    static std::shared_ptr<Tensor<D, R>> linear(
            const std::shared_ptr<Tensor<D, R>> &w,
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &b) {

        if (w->data->dimension(0) != x->data->dimension(0))
            throw std::invalid_argument("shapes of w and x mismatch");
        if (b != nullptr && w->data->dimension(1) != b->data->dimension(0))
            throw std::invalid_argument("shapes of w and b mismatch");


       std::array<std::int64_t, R> shape {w->data->dimension(1), x->data->dimension(1)};

        auto t = w->data->contract(*x->data, Eigen::array<Eigen::IndexPair <std::int64_t>, 1>{Eigen::IndexPair <std::int64_t>(0, 0)});
        std::shared_ptr<Tensor<D, R>> result;

        if (b != nullptr) {
            Eigen::array<std::int64_t, R> reshape {shape[0], 1};
            Eigen::array<std::int64_t, R> broadcast {1, shape[1]};
            result = std::make_shared<Tensor<D, R>>(t + b->data->reshape(reshape).broadcast(broadcast), shape);
        } else
            result = std::make_shared<Tensor<D, R>>(t, shape);

        if ((w->needsGradient() || x->needsGradient() || (b != nullptr && b->needsGradient())) && !CNodeBase::noGrad) {
            if (b != nullptr)
                result->setGradFn(std::make_shared<Linear<D>>(w, x, b, result));
            else
                result->setGradFn(std::make_shared<Linear<D>>(w, x, result));
        }
        return result;
    }

    void computeGradients() override {
        if (cw.has_value())
            cw.value()->addGrad(x->contract(*CNode<D, R>::grad, Eigen::array<Eigen::IndexPair <std::int64_t>, 1>{Eigen::IndexPair <std::int64_t>(1, 1)}));
        if (cx.has_value())
            cx.value()->addGrad(w->contract(*CNode<D, R>::grad, Eigen::array<Eigen::IndexPair <std::int64_t>, 1>{Eigen::IndexPair <std::int64_t>(1, 0)}));
        if (cb.has_value())
            cb.value()->addGrad(CNode<D, R>::grad->sum(Eigen::array <std::int64_t, 1> {1}));
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::Tensor<D, R>> w;
    std::shared_ptr<Eigen::Tensor<D, R>> x;
    std::optional<std::shared_ptr<CNode<D, R>>> cw;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::optional<std::shared_ptr<CNode<D, 1>>> cb;
};

#undef R

#endif //LIBDL_LINEAR_H
