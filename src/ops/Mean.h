
#ifndef LIBDL_MEAN_H
#define LIBDL_MEAN_H


#include "../Tensor.h"
#include "../Utils.h"


template <typename D, int R>
class Mean : public CNode<D, 0> {
public:
    Mean(
            const std::optional<std::shared_ptr<CNode<D, R>>> &cx,
            const std::shared_ptr<Tensor<D, 0>> &result,
            const std::array<long, R> &shape)
            : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({cx}), result),
            cx(cx),
            shape(shape) {}

    /*
     * \brief computes the mean along all axes
     *
     * \param x a tensor of any shape for which the mean should be computed
     *
     * \return a 0d tensor containing the mean
     * */
    static std::shared_ptr<Tensor<D, 0>> mean(
            const std::shared_ptr<Tensor<D, R>> &x) {

        auto result = std::make_shared<Tensor<D, 0>>(x->data->mean(), std::array<long, 0> {});
        if (x->needsGradient())
            result->setGradFn(std::make_shared<Mean<D, R>>(x->gradFn, result, x->data->dimensions()));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            std::array<long, R> r;
            r.fill(1);
            auto t = CNode<D, 0>::grad->reshape(r).broadcast(shape);
            long size = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<>());
            cx.value()->addGrad(t / t.constant(size));
        }
        CNode<D, 0>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::array<long, R> shape;
};


#endif //LIBDL_MEAN_H
