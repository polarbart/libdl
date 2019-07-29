
#ifndef LIBDL_BATCHNORM2D_H
#define LIBDL_BATCHNORM2D_H


#include "CNode.h"
#include "../Utils.h"

#define R 4

template<typename D>
class BatchNorm2D : public CNode<D, R> {

public:
    BatchNorm2D(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &gamma,
            const std::shared_ptr<Tensor<D, 1>> &beta,
            Eigen::Tensor<D, 1> mean,
            Eigen::Tensor<D, 1> var,
            Eigen::Tensor<D, R> xh,
            D epsilon,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, gamma->gradFn, beta->gradFn}), result),
            cx(x->gradFn),
            cgamma(gamma->gradFn),
            cbeta(beta->gradFn),
            x(x->data),
            gamma(gamma->data),
            mean(std::move(mean)),
            var(std::move(var)),
            epsilon(epsilon),
            useRunningAvgVar(false),
            xh(std::move(xh)) {}

    BatchNorm2D(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &gamma,
            const std::shared_ptr<Tensor<D, 1>> &beta,
            const std::shared_ptr<Tensor<D, 1>> &runningVar,
            Eigen::Tensor<D, R> xh,
            D epsilon,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, gamma->gradFn, beta->gradFn}), result),
            cx(x->gradFn),
            cgamma(gamma->gradFn),
            cbeta(beta->gradFn),
            x(x->data),
            gamma(gamma->data),
            runningVar(runningVar->data),
            epsilon(epsilon),
            useRunningAvgVar(true),
            xh(std::move(xh)) {}

    /*
     * \brief performs batchnorm on the image like input
     *
     * \param x a 4d tensor on which batchnorm should be performed with shape (channels, width, height, batchsize)
     * \param gamma the gamma parameter of batchnorm with shape (channels,)
     * \param beta the beta parameter of batchnorm with shape (channels,)
     * \param runningMean running mean of x with shape (channels,)
     * \param runningVar running variance of x with shape (channels,)
     * \param momentum the momentum parameter of batchnorm
     * \param epsilon the epsilon parameter of batchnorm
     * \param useRunningAvgVar if running average and variance should be used
     *
     * \return a new normalized tensor with the same shape as x
     * */

    static std::shared_ptr<Tensor<D, R>> batchNorm2d(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, 1>> &gamma,
            const std::shared_ptr<Tensor<D, 1>> &beta,
            const std::shared_ptr<Tensor<D, 1>> &runningMean,
            const std::shared_ptr<Tensor<D, 1>> &runningVar,
            D momentum,
            D epsilon,
            bool useRunningAvgVar) {

        if (gamma->data->dimension(0) != x->data->dimension(0))
            throw std::invalid_argument("gamma has wrong shape");
        if (beta->data->dimension(0) != x->data->dimension(0))
            throw std::invalid_argument("beta has wrong shape");
        if (runningMean->data->dimension(0) != x->data->dimension(0))
            throw std::invalid_argument("runningMean has wrong shape");
        if (runningVar->data->dimension(0) != x->data->dimension(0))
            throw std::invalid_argument("runningVar has wrong shape");
        if (momentum < 0)
            throw std::invalid_argument("momentum must not be negative");
        if (epsilon < 0)
            throw std::invalid_argument("epsilon must not be negative");

        const Eigen::array<long, R> reshape{x->data->dimension(0), 1, 1, 1};
        const Eigen::array<long, R> broadcast{1, x->data->dimension(1), x->data->dimension(2), x->data->dimension(3)};
        const Eigen::array<int, 3> meanDims{1, 2, 3};

        std::shared_ptr<Tensor<D, R>> result;

        // https://wiki.tum.de/display/lfdv/Batch+Normalization
        if (useRunningAvgVar) {
            Eigen::Tensor<D, R> xh(x->data->dimensions());
            xh.device(GlobalThreadPool::myDevice) = (*x->data - runningMean->data->reshape(reshape).broadcast(broadcast)) /
                                  ((*runningVar->data + runningVar->data->constant(epsilon)).sqrt().eval().reshape(reshape).broadcast(broadcast));
            auto y = gamma->data->reshape(reshape).broadcast(broadcast) * xh + beta->data->reshape(reshape).broadcast(broadcast);

            result = std::make_shared<Tensor<D, R>>(y, x->data->dimensions());

            if ((x->needsGradient() || gamma->needsGradient() || beta->needsGradient()) && !CNodeBase::noGrad)
                result->setGradFn(std::make_shared<BatchNorm2D<D>>(x, gamma, beta, runningVar, std::move(xh), epsilon, result));

        } else {
            Eigen::Tensor<D, 1> mean(x->data->dimension(0)), var(x->data->dimension(0));
            mean.device(GlobalThreadPool::myDevice) = x->data->mean(meanDims);
            auto xm = (*x->data - mean.reshape(reshape).broadcast(broadcast)).eval();
            var.device(GlobalThreadPool::myDevice) = xm.square().mean(meanDims);

            Eigen::Tensor<D, R> xh(x->data->dimensions());
            xh.device(GlobalThreadPool::myDevice) = xm / ((var + var.constant(epsilon)).sqrt().eval().reshape(reshape).broadcast(broadcast));
            auto y = gamma->data->reshape(reshape).broadcast(broadcast) * xh + beta->data->reshape(reshape).broadcast(broadcast);

            result = std::make_shared<Tensor<D, R>>(y, x->data->dimensions());

            runningMean->data->device(GlobalThreadPool::myDevice) = mean.constant(momentum) * mean + runningMean->data->constant(1 - momentum) * *runningMean->data;
            runningVar->data->device(GlobalThreadPool::myDevice) = var.constant(momentum) * var + runningVar->data->constant(1 - momentum) * *runningVar->data;

            if ((x->needsGradient() || gamma->needsGradient() || beta->needsGradient()) && !CNodeBase::noGrad)
                result->setGradFn(std::make_shared<BatchNorm2D<D>>(x, gamma, beta, std::move(mean), std::move(var), std::move(xh), epsilon, result));

        }
        return result;
    }

    void computeGradients() override {
        // #efficient
        const Eigen::array<long, R> reshape{x->dimension(0), 1, 1, 1};
        const Eigen::array<long, R> broadcast{1, x->dimension(1), x->dimension(2), x->dimension(3)};
        const Eigen::array<int, 3> meanDims{1, 2, 3};
        if (cx.has_value()) {
            if (useRunningAvgVar) {
                cx.value()->addGrad((*gamma / (*runningVar + runningVar->constant(epsilon)).sqrt()).eval().reshape(reshape).broadcast(broadcast) * *CNode<D, R>::grad);
            } else {
                auto rvpe = (var + var.constant(epsilon)).sqrt().eval();
                auto xmm = (*x - mean.reshape(reshape).broadcast(broadcast)).eval();
                int m = x->dimension(1) * x->dimension(2) * x->dimension(3);

                auto dxh = (gamma->reshape(reshape).broadcast(broadcast) * *CNode<D, R>::grad).eval();
                auto dv = ((dxh * xmm).sum(meanDims) * var.constant(-.5) / rvpe.cube()).eval();
                // auto dm = -dxh.sum(meanDims) / rvpe + dv * xmm.mean(meanDims) * dv.constant(-2);
                auto dx = -dxh / rvpe.reshape(reshape).broadcast(broadcast) + dv.reshape(reshape).broadcast(broadcast) * (xmm * xmm.constant(2. / m) + xmm.constant(1. / m));
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

    bool useRunningAvgVar;
    D epsilon;

    std::shared_ptr<Eigen::Tensor<D, R>> x;
    std::shared_ptr<Eigen::Tensor<D, 1>> gamma;
    Eigen::Tensor<D, 1> mean;
    Eigen::Tensor<D, 1> var;
    std::shared_ptr<Eigen::Tensor<D, 1>> runningVar;
    Eigen::Tensor<D, R> xh;


};

#undef R

#endif //LIBDL_BATCHNORM2D_H
