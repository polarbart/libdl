
#ifndef LIBDL_LEAKYRELU_H
#define LIBDL_LEAKYRELU_H

#include "CNode.h"
#include "../Utils.h"


template <typename D, int R>
class LeakyRelu : public CNode<D, R> {

public:
    LeakyRelu(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &result,
            D negativeSlope)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result),
            x(x->data),
            cx(x->gradFn),
            negativeSlope(negativeSlope){}

    /*
     * \brief applies the leaky relu function elementwise
     *
     * \param x tensor of any shape
     * \param negativeSlope factor by which negative values are scaled
     *
     * \return a new tensor with the same shape as x in which all negative values are scaled by negativeSlope
     * */
    static std::shared_ptr<Tensor<D, R>> leakyRelu(
            const std::shared_ptr<Tensor<D, R>> &x,
            D negativeSlope) {

        auto mask = (*x->data >= x->data->constant(0)).select(x->data->constant(1), x->data->constant(negativeSlope));
        auto result = std::make_shared<Tensor<D, R>>(*x->data * mask, x->data->dimensions());

        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<LeakyRelu<D, R>>(x, result, negativeSlope));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            auto mask = (*x >= x->constant(0)).select(x->constant(1), x->constant(negativeSlope));
            cx.value()->addGrad(mask * *CNode<D, R>::grad);
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::Tensor<D, R>> x;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    D negativeSlope;


};

#endif //LIBDL_LEAKYRELU_H
