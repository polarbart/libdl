//
// Created by superbabes on 16.06.19.
//

#ifndef LIBDL_CROSSENTROPYWITHLOGITS_H
#define LIBDL_CROSSENTROPYWITHLOGITS_H


#include "CNode.h"
#include "Utils.h"

template <typename D, int R>
class CrossEntropyWithLogits : public CNode<D, 0> {

public:
    template <typename DerivedOther>
    CrossEntropyWithLogits(
            const DerivedOther &softmax,
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &y,
            const std::shared_ptr<Tensor<D, 0>> &result) : CNode<D, 0>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, y->gradFn}), result), cx(x->gradFn), cy(y->gradFn), softmax(softmax), y(y->eTensor) {}

    static std::shared_ptr<Tensor<D, 0>> crossEntropyWithLogits(const std::shared_ptr<Tensor<D, R>> &x, const std::shared_ptr<Tensor<D, R>> &y) {
        static Eigen::ThreadPool pool(8);
        static Eigen::ThreadPoolDevice myDevice(&pool, 8);

        Eigen::array<long, R> reshape = x->eTensor->dimensions();
        reshape[0] = 1;
        Eigen::array<long, R> broadcast;
        broadcast.fill(1);
        broadcast[0] = x->eTensor->dimension(0);
        auto i1 = (*x->eTensor - x->eTensor->maximum(Eigen::array<int, 1> {0}).eval().reshape(reshape).broadcast(broadcast)).exp();

        Eigen::Tensor<D, R> softmax(x->eTensor->dimensions());
        softmax.device(myDevice) = i1 / i1.sum(Eigen::array<int, 1> {0}).eval().reshape(reshape).broadcast(broadcast) + i1.constant(1e-8);
        auto mce = (-softmax.log() * *y->eTensor).mean();
        auto result = std::make_shared<Tensor<D, 0>>(mce * mce.constant(x->eTensor->dimension(0)), std::array<long, 0> {});
        if (x->needsGradient())
            result->setGradFn(std::make_shared<CrossEntropyWithLogits<D, R>>(softmax, x, y, result));
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
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> y;

};

#endif //LIBDL_CROSSENTROPYWITHLOGITS_H
