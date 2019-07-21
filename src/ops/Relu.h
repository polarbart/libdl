
#ifndef LIBDL_RELU_H
#define LIBDL_RELU_H


#include "CNode.h"
#include "../Utils.h"

template <typename D, int R>
class Relu : public CNode<D, R> {

public:
    Relu(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result),
            x(x->eTensor),
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
        auto tmp = (*x->eTensor >= x->eTensor->constant(0)).select(*x->eTensor, x->eTensor->constant(0));
        auto result = std::make_shared<Tensor<D, R>>(tmp, x->eTensor->dimensions());
        if (x->needsGradient())
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
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> x;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;


};


#endif //LIBDL_RELU_H
