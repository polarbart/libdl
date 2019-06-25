//
// Created by superbabes on 20.06.19.
//

#ifndef LIBDL_BATCHNORM2D_H
#define LIBDL_BATCHNORM2D_H


#include "CNode.h"
#include "Utils.h"

#define R 4
// https://wiki.tum.de/display/lfdv/Batch+Normalization
template <typename D>
class BatchNorm2D : public CNode<D, R> {

public:
    template <typename DerivedOther, typename DerivedOther2, typename DerivedOther3>
    BatchNorm2D(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &gamma,
            const std::shared_ptr<Tensor<D, 1>> &beta,
            const DerivedOther &mean,
            const DerivedOther2 &var,
            const DerivedOther3 &xh,
            D epsilon,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, gamma->gradFn, beta->gradFn}), result),
            cx(x->gradFn),
            cgamma(gamma->gradFn),
            cbeta(beta->gradFn),
            x(x->eTensor),
            gamma(gamma->eTensor),
            mean(mean),
            var(var),
            epsilon(epsilon),
            useRunningAvg(false),
            xh(xh){}

    template <typename DerivedOther>
    BatchNorm2D(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &gamma,
            const std::shared_ptr<Tensor<D, 1>> &beta,
            const std::shared_ptr<Tensor<D, 1>> &runningVar,
            const DerivedOther &xh,
            D epsilon,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, gamma->gradFn, beta->gradFn}), result),
              cx(x->gradFn),
              cgamma(gamma->gradFn),
              cbeta(beta->gradFn),
              x(x->eTensor),
              gamma(gamma->eTensor),
              runningVar(runningVar->eTensor),
              epsilon(epsilon),
              useRunningAvg(true),
              xh(xh){}

    static std::shared_ptr<Tensor<D, R>> batchNorm2d(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &gamma,
            const std::shared_ptr<Tensor<D, 1>> &beta,
            const std::shared_ptr<Tensor<D, 1>> &runningMean,
            const std::shared_ptr<Tensor<D, 1>> &runningVar,
            D momentum,
            D epsilon,
            bool useRunningAvg) {

        const Eigen::array<long, R> reshape {1, 1, x->eTensor->dimension(2), 1};
        const Eigen::array<long, R> broadcast {x->eTensor->dimension(0), x->eTensor->dimension(1), 1, x->eTensor->dimension(3)};
        const Eigen::array<int, 3> meanDims {0, 1, 3};

        std::shared_ptr<Tensor<D, R>> result;

        if (useRunningAvg) {
            auto xh = (*x->eTensor - runningMean->eTensor->reshape(reshape).broadcast(broadcast)) / ((*runningVar->eTensor + runningVar->eTensor->constant(epsilon)).sqrt().reshape(reshape).broadcast(broadcast));
            auto y = gamma->eTensor->reshape(reshape).broadcast(broadcast) * xh + beta->eTensor->reshape(reshape).broadcast(broadcast);

            result = std::make_shared<Tensor<D, R>>(y, x->eTensor->dimensions());

            if (x->needsGradient() || gamma->needsGradient() || beta->needsGradient())
                result->setGradFn(std::make_shared<BatchNorm2D<D>>(x, gamma, beta, runningVar, xh, epsilon, result));

        } else {

            auto mean = x->eTensor->mean(meanDims);
            auto var = (*x->eTensor - mean.reshape(reshape).broadcast(broadcast)).square().mean(meanDims);

            auto xh = (*x->eTensor - mean.reshape(reshape).broadcast(broadcast)) / ((var + var.constant(epsilon)).sqrt().reshape(reshape).broadcast(broadcast));
            auto y = gamma->eTensor->reshape(reshape).broadcast(broadcast) * xh + beta->eTensor->reshape(reshape).broadcast(broadcast);

            result = std::make_shared<Tensor<D, R>>(y, x->eTensor->dimensions());

            *runningMean->eTensor = mean.constant(momentum) * mean + runningMean->eTensor->constant(momentum) * *runningMean->eTensor;
            *runningVar->eTensor = var.constant(momentum) * var + runningVar->eTensor->constant(momentum) * *runningVar->eTensor;

            if (x->needsGradient() || gamma->needsGradient() || beta->needsGradient())
                result->setGradFn(std::make_shared<BatchNorm2D<D>>(x, gamma, beta, mean, var, xh, epsilon, result));

        }
        return result;
    }

    void computeGradients() override {
        const Eigen::array<long, R> reshape {1, 1, x->dimension(2), 1};
        const Eigen::array<long, R> broadcast {x->dimension(0), x->dimension(1), 1, x->dimension(3)};
        const Eigen::array<int, 3> meanDims {0, 1, 3};
        if (cx.has_value()) {
            if (useRunningAvg) {
                cx.value()->addGrad((*gamma / *runningVar).broadcast(broadcast).reshape(reshape) * *CNode<D, R>::grad);
            } else {
                auto rvpe = (var + var.constant(epsilon)).sqrt();
                auto xmm = *x - mean.reshape(reshape).broadcast(broadcast);
                int m = x->dimension(0) * x->dimension(1) * x->dimension(3);

                auto dxh = gamma->reshape(reshape).broadcast(broadcast) * *CNode<D, R>::grad;
                auto dv = (dxh * xmm).sum(meanDims) * var.constant(-.5) / rvpe.pow(3);
                auto dm = -dxh.sum(meanDims) / rvpe + dv * xmm.mean(meanDims) * dv.constant(-2);
                auto dx = -dxh / rvpe.reshape(reshape).broadcast(broadcast)
                        + dv.reshape(reshape).broadcast(broadcast) * (xmm * xmm.constant(2./m) + xmm.constant(1./m));
                cx.value()->addGrad(dx);
            }
        }
        if (cgamma.has_value())
            cgamma.value()->addGrad((xh * *CNode<D, R>::grad).sum(meanDims));
        if (cbeta.has_value())
            cbeta.value()->addGrad(CNode<D, R>::grad->sum(meanDims));
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::optional<std::shared_ptr<CNode<D, 1>>> cgamma;
    std::optional<std::shared_ptr<CNode<D, 1>>> cbeta;

    bool useRunningAvg;
    D epsilon;

    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> x;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, 1>>> gamma;
    Eigen::Tensor<D, 1> mean;
    Eigen::Tensor<D, 1> var;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, 1>>> runningVar;
    Eigen::Tensor<D, R> xh;


};

#undef R

#endif //LIBDL_BATCHNORM2D_H
