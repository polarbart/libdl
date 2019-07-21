
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
            x(x->eTensor),
            gamma(gamma->eTensor),
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
            x(x->eTensor),
            gamma(gamma->eTensor),
            runningVar(runningVar->eTensor),
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

        if (gamma->eTensor->dimension(0) != x->eTensor->dimension(0))
            throw std::invalid_argument("gamma has wrong shape");
        if (beta->eTensor->dimension(0) != x->eTensor->dimension(0))
            throw std::invalid_argument("beta has wrong shape");
        if (runningMean->eTensor->dimension(0) != x->eTensor->dimension(0))
            throw std::invalid_argument("runningMean has wrong shape");
        if (runningVar->eTensor->dimension(0) != x->eTensor->dimension(0))
            throw std::invalid_argument("runningVar has wrong shape");
        if (momentum < 0)
            throw std::invalid_argument("momentum must not be negative");
        if (epsilon < 0)
            throw std::invalid_argument("epsilon must not be negative");

        const Eigen::array<long, R> reshape{x->eTensor->dimension(0), 1, 1, 1};
        const Eigen::array<long, R> broadcast{1, x->eTensor->dimension(1), x->eTensor->dimension(2), x->eTensor->dimension(3)};
        const Eigen::array<int, 3> meanDims{1, 2, 3};

        static Eigen::ThreadPool pool(8);
        static Eigen::ThreadPoolDevice myDevice(&pool, 8);

        std::shared_ptr<Tensor<D, R>> result;

        // https://wiki.tum.de/display/lfdv/Batch+Normalization
        if (useRunningAvgVar) {
            Eigen::Tensor<D, R> xh(x->eTensor->dimensions());
            xh.device(myDevice) = (*x->eTensor - runningMean->eTensor->reshape(reshape).broadcast(broadcast)) /
                                  ((*runningVar->eTensor + runningVar->eTensor->constant(epsilon)).sqrt().eval().reshape(reshape).broadcast(broadcast));
            auto y = gamma->eTensor->reshape(reshape).broadcast(broadcast) * xh + beta->eTensor->reshape(reshape).broadcast(broadcast);

            result = std::make_shared<Tensor<D, R>>(y, x->eTensor->dimensions());

            if (x->needsGradient() || gamma->needsGradient() || beta->needsGradient())
                result->setGradFn(std::make_shared<BatchNorm2D<D>>(x, gamma, beta, runningVar, std::move(xh), epsilon, result));

        } else {
            Eigen::Tensor<D, 1> mean(x->eTensor->dimension(0)), var(x->eTensor->dimension(0));
            mean.device(myDevice) = x->eTensor->mean(meanDims);
            auto xm = (*x->eTensor - mean.reshape(reshape).broadcast(broadcast)).eval();
            var.device(myDevice) = xm.square().mean(meanDims);

            Eigen::Tensor<D, R> xh(x->eTensor->dimensions());
            xh.device(myDevice) = xm / ((var + var.constant(epsilon)).sqrt().eval().reshape(reshape).broadcast(broadcast));
            auto y = gamma->eTensor->reshape(reshape).broadcast(broadcast) * xh + beta->eTensor->reshape(reshape).broadcast(broadcast);

            result = std::make_shared<Tensor<D, R>>(y, x->eTensor->dimensions());

            runningMean->eTensor->device(myDevice) = mean.constant(momentum) * mean + runningMean->eTensor->constant(1 - momentum) * *runningMean->eTensor;
            runningVar->eTensor->device(myDevice) = var.constant(momentum) * var + runningVar->eTensor->constant(1 - momentum) * *runningVar->eTensor;

            if (x->needsGradient() || gamma->needsGradient() || beta->needsGradient())
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

    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> x;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, 1>>> gamma;
    Eigen::Tensor<D, 1> mean;
    Eigen::Tensor<D, 1> var;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, 1>>> runningVar;
    Eigen::Tensor<D, R> xh;


};

#undef R

#endif //LIBDL_BATCHNORM2D_H
