
#ifndef LIBDL_POW_H
#define LIBDL_POW_H

#include "../Tensor.h"
#include "../Utils.h"

template <typename D, int R>
class Pow : public CNode<D, R> {
public:
    Pow(
            const std::shared_ptr<Tensor<D, R>> &x,
            D p,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result),
            x(x->data),
            cx(x->gradFn),
            p(p) {}

    /*
     * \brief computes x to the power of p elementwise
     *
     * \param x the tensor for which the power should be computed
     * \param p the power to which the tensor should be raised
     *
     * \return a new tensor with the same shape as x in which all elements have been raised to the power of p
     * */
    static std::shared_ptr<Tensor<D, R>> pow(
            const std::shared_ptr<Tensor<D, R>> &x,
            D p) {

        auto result = std::make_shared<Tensor<D, R>>(x->data->pow(p), x->data->dimensions());
        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<Pow<D, R>>(x, p, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value())
            cx.value()->addGrad(x->constant(p) * x->pow(p - 1) * *CNode<D, R>::grad);
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::Tensor<D, R>> x;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    D p;
};


#endif //LIBDL_POW_H
