
#ifndef LIBDL_CROSSENTROPYWITHLOGITS_H
#define LIBDL_CROSSENTROPYWITHLOGITS_H


#include "CNode.h"
#include "../Utils.h"

#define R 2

template <typename D>
class CrossEntropyWithLogits : public CNode<D, 0> {

public:

    CrossEntropyWithLogits(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &y,
            Eigen::Tensor<D, R> softmax,
            const std::shared_ptr<Tensor<D, 0>> &result)
            : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, y->gradFn}), result),
            cx(x->gradFn),
            cy(y->gradFn),
            softmax(std::move(softmax)),
            y(y->data) {}

    /*
     * \brief apply softmax to x along the zero'th dimension
     *        then compute the mean cross entropy between x and y
     *
     * \param x two dimensional tensor on which softmax is applied
     * \param y one hot encoded tensor with the same shape as x
     *
     * \return a zero dimensional tensor containing the cross entropy
     * */
    static std::shared_ptr<Tensor<D, 0>> crossEntropyWithLogits(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &y) {

        for (std::int64_t i = 0; i < R; i++)
            if (x->data->dimension(i) != y->data->dimension(i))
                throw std::invalid_argument("the shapes of x and y must match");

        #pragma omp parallel for
        for (std::int64_t i = 0; i < y->data->dimension(1); i++) {
            bool hasOne = false;
            for (std::int64_t j = 0; j < y->data->dimension(0); j++) {
                if ((*y->data)(j, i) == 1) {
                    if (hasOne)
                        throw std::invalid_argument("y is not one hot encoded");
                    else
                        hasOne = true;
                } else if ((*y->data)(j, i) != 0)
                    throw std::invalid_argument("y is not one hot encoded");
            }
            if (!hasOne)
                throw std::invalid_argument("y is not one hot encoded");
        }

        Eigen::array<std::int64_t, R> reshape = x->data->dimensions();
        reshape[0] = 1;
        Eigen::array<std::int64_t, R> broadcast;
        broadcast.fill(1);
        broadcast[0] = x->data->dimension(0);
        auto i1 = (*x->data - x->data->maximum(Eigen::array <std::int64_t, 1> {0}).eval().reshape(reshape).broadcast(broadcast)).exp();

        Eigen::Tensor<D, R> softmax(x->data->dimensions());
        softmax.device(GlobalThreadPool::myDevice) = i1 / i1.sum(Eigen::array <std::int64_t, 1> {0}).eval().reshape(reshape).broadcast(broadcast) + i1.constant(1e-32f);
        auto mce = (-softmax.log() * *y->data).mean();
        auto result = std::make_shared<Tensor<D, 0>>(mce * mce.constant(static_cast<std::float_t>(x->data->dimension(0))), std::array<std::int64_t, 0> {});
        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<CrossEntropyWithLogits<D>>(x, y, std::move(softmax), result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            cx.value()->addGrad((softmax - *y) * softmax.constant((*CNode<D, 0>::grad)(0) / softmax.dimension(1)));
        }
        if (cy.has_value()) {
            cy.value()->addGrad(-softmax.log() * softmax.constant((*CNode<D, 0>::grad)(0) / softmax.dimension(1)));
        }
        CNode<D, 0>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::optional<std::shared_ptr<CNode<D, R>>> cy;
    Eigen::Tensor<D, R> softmax;
    std::shared_ptr<Eigen::Tensor<D, R>> y;

};

#undef R

#endif //LIBDL_CROSSENTROPYWITHLOGITS_H
