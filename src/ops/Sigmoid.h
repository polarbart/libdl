
#ifndef LIBDL_SIGMOID_H
#define LIBDL_SIGMOID_H

#include "../Tensor.h"
#include "../Utils.h"

template <typename D, int R>
class Sigmoid : public CNode<D, R> {
public:
    Sigmoid(
            const std::optional<std::shared_ptr<CNode<D, R>>> &cx,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({cx}), result),
            cx(cx),
            result(result->data) {}

    /*
     * \brief applies the sigmoid function elementwise
     *
     * \param x tensor of any shape
     *
     * \return a new tensor with the same shape as x
     * */
    static std::shared_ptr<Tensor<D, R>> sigmoid(
            const std::shared_ptr<Tensor<D, R>> &x) {
        auto result = std::make_shared<Tensor<D, R>>(x->data->sigmoid(), x->data->dimensions());
        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<Sigmoid<D, R>>(x->gradFn, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value())
            cx.value()->addGrad(*result * (result->constant(1) - *result) * *CNode<D, R>::grad);
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::shared_ptr<Eigen::Tensor<D, R>> result;
};


#endif //LIBDL_SIGMOID_H
