
#ifndef LIBDL_RELU_H
#define LIBDL_RELU_H


#include "CNode.h"
#include "../Utils.h"

template <typename D, std::int64_t R>
class Relu : public CNode<D, R> {

public:
    Relu(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result),
            x(x->data),
            cx(x->gradFn) {}

    /*
     * \brief applies the relu function elementwise
     *
     * \param x tensor of any shape
     *
     * \return a new tensor with the same shape as x in which all negative values are set to zero
     * */
    static std::shared_ptr<Tensor<D, R>> relu(
            const std::shared_ptr<Tensor<D, R>> &x) {
        auto tmp = (*x->data >= x->data->constant(0)).select(*x->data, x->data->constant(0));
        auto result = std::make_shared<Tensor<D, R>>(tmp, x->data->dimensions());
        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<Relu<D, R>>(x, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            auto grad = (*x >= x->constant(0)).select(*CNode<D, R>::grad, x->constant(0));
            cx.value()->addGrad(grad);
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::Tensor<D, R>> x;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;


};


#endif //LIBDL_RELU_H
