
#include <catch2/catch.hpp>
#include "../../src/ops/BatchNorm2D.h"
#include "../Helper.h"

TEST_CASE("batchnorm2d") {
    auto x = trange<4>({4, 8, 8, 2}, true, 10);
    auto gamma = trange<1>({4});
    auto beta = trange<1>({4});
    auto runningMean = constant<1>({4}, 0);
    auto runningVar = constant<1>({4}, 1);

    Eigen::array<std::int64_t, 3> sum{1, 2, 3};
    Eigen::array<std::int64_t, 4> reshape{x->data->dimension(0), 1, 1, 1};
    Eigen::array<std::int64_t, 4> broadcast{1, x->data->dimension(1), x->data->dimension(2), x->data->dimension(3)};

    SECTION("useRunningAvgVar = false") {
        auto r = BatchNorm2D<std::float_t>::batchNorm2d(x, gamma, beta, runningMean, runningVar, 0.9, 1e-5, false);

        auto grad = setGradAndBackward<4>(r, 50);

        Eigen::Tensor<std::float_t, 1> mean = x->data->mean(sum);
        auto xm = (*x->data - mean.reshape(reshape).broadcast(broadcast)).eval();
        Eigen::Tensor<std::float_t, 1> var = xm.square().mean(sum);
        auto xh = (xm / (var + var.constant(1e-5)).sqrt().eval().reshape(reshape).broadcast(broadcast)).eval();
        Eigen::Tensor<std::float_t, 4> rReference = gamma->data->reshape(reshape).broadcast(broadcast) * xh +
                                             beta->data->reshape(reshape).broadcast(broadcast);

        REQUIRE(tensorEqual(*r->data, rReference));
        REQUIRE(tensorEqual<1>(*runningMean->data, mean * mean.constant(0.9)));
        REQUIRE(tensorEqual<1>(*runningVar->data, var * var.constant(0.9) + var.constant(0.1)));

        // check if mean and std of r is equal to beta and gamma
        REQUIRE(tensorEqual<1>(r->data->mean(sum), *beta->data, 1e-4));
        REQUIRE(tensorEqual<1>((*r->data - beta->data->reshape(reshape).broadcast(broadcast)).eval().square().mean(sum).sqrt(), *gamma->data, 1e-4));

        // gradient checks
        REQUIRE(tensorEqual<1>(*gamma->grad, (xh * grad).sum(sum)));
        REQUIRE(tensorEqual<1>(*beta->grad, grad.sum(sum)));

        auto rvpe = (var + var.constant(1e-5)).sqrt().eval();
        auto xmm = (*x->data - mean.reshape(reshape).broadcast(broadcast)).eval();

        auto dxh = (gamma->data->reshape(reshape).broadcast(broadcast) * grad).eval();
        auto dv = ((dxh * xmm).sum(sum) * var.constant(-.5) / rvpe.cube()).eval();
        Eigen::Tensor<std::float_t, 4> dx = -dxh / rvpe.reshape(reshape).broadcast(broadcast) + dv.reshape(reshape).broadcast(broadcast) * (xmm * xmm.constant(2. / 128) + xmm.constant(1. / 128));

        REQUIRE(tensorEqual(*x->grad, dx));
    }
    SECTION("useRunningAvgVar = true") {
        auto r = BatchNorm2D<std::float_t>::batchNorm2d(x, gamma, beta, runningMean, runningVar, 0.9, 0, true);

        auto grad = setGradAndBackward<4>(r, 10);

        Eigen::Tensor<std::float_t, 4> rReference = gamma->data->reshape(reshape).broadcast(broadcast) * *x->data +
                                             beta->data->reshape(reshape).broadcast(broadcast);

        REQUIRE(tensorEqual(*r->data, rReference, 1e-4));
        REQUIRE(tensorEqual<1>(*runningMean->data, runningMean->data->constant(0)));
        REQUIRE(tensorEqual<1>(*runningVar->data, runningVar->data->constant(1)));

        // gradient checks
        REQUIRE(tensorEqual<1>(*gamma->grad, (*x->data * grad).sum(sum), 1e-3));
        REQUIRE(tensorEqual<1>(*beta->grad, grad.sum(sum)));
        REQUIRE(tensorEqual<4>(*x->grad, gamma->data->reshape(reshape).broadcast(broadcast) * grad, 1e-4));
    }
}