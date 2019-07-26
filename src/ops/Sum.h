
#ifndef LIBDL_SUM_H
#define LIBDL_SUM_H


#include "../Tensor.h"
#include "../Utils.h"

template <typename D, int R>
class Sum : public CNode<D, 0> {
public:
    Sum(
            const std::optional<std::shared_ptr<CNode<D, R>>> &cx,
            const std::shared_ptr<Tensor<D, 0>> &result,
            const std::array<long, R> &shape)
            : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({cx}), result),
            cx(cx),
            shape(shape) {}

    /*
     * \brief computes the sum along all axes
     *
     * \param x a tensor of any shape for which the sum should be computed
     *
     * \return a 0d tensor containing the sum
     * */
    static std::shared_ptr<Tensor<D, 0>> sum(
            const std::shared_ptr<Tensor<D, R>> &x) {
        auto result = std::make_shared<Tensor<D, 0>>(x->data->sum(), std::array<long, 0> {});
        if (x->needsGradient())
            result->setGradFn(std::make_shared<Sum<D, R>>(x->gradFn, result, x->data->dimensions()));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            std::array<int, R> r;
            r.fill(1);
            cx.value()->addGrad(CNode<D, 0>::grad->reshape(r).broadcast(shape));
        }
        CNode<D, 0>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::array<long, R> shape;
};

#endif //LIBDL_SUM_H
