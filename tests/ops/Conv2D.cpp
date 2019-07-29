
#include <catch2/catch.hpp>
#include "../../src/ops/Conv2D.h"
#include "../Helper.h"

Eigen::Tensor<float, 4> conv(
        const Eigen::Tensor<float, 4> &x,
        const Eigen::Tensor<float, 4> &f,
        const Eigen::Tensor<float, 1> &b,
        int padding,
        int stride) {

    std::array<long, 4> newDims{
            f.dimension(3),
            (x.dimension(1) - f.dimension(1) + 2 * padding) / stride + 1,
            (x.dimension(2) - f.dimension(2) + 2 * padding) / stride + 1,
            x.dimension(3)
    };

    Eigen::array<Eigen::IndexPair<int>, 4> ePadding {
            Eigen::IndexPair<int>(0, 0),
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(0, 0)
    };
    Eigen::Tensor<float, 4> padded = x.pad(ePadding);

    Eigen::Tensor<float, 4> result(newDims);

    for (int n = 0; n < x.dimension(3); n++)
        for (int w = 0; w < newDims[2]; w++)
            for (int h = 0; h < newDims[1]; h++)
                for (int cf = 0; cf < f.dimension(3); cf++) {
                    float sum = 0;
                    for (int cw = 0; cw < f.dimension(2); cw++)
                        for (int ch = 0; ch < f.dimension(1); ch++)
                            for (int c = 0; c < f.dimension(0); c++)
                                sum += f(c, ch, cw, cf) * padded(c, stride * h + ch, stride * w + cw, n);
                    result(cf, h, w, n) = sum + b(cf);

                }
    return result;

}

template <int N>
void testGradientNumerically(
        const std::shared_ptr<Tensor<float, N>> &t,
        const std::shared_ptr<Tensor<float, 4>> &x,
        const std::shared_ptr<Tensor<float, 4>> &f,
        const std::shared_ptr<Tensor<float, 1>> &b,
        const std::shared_ptr<Tensor<float, 4>> &r,
        int padding,
        int stride) {

    for (int i = 0; i < t->data->size(); i++) {
        float oldValue = t->data->data()[i];
        t->data->data()[i] += 1;

        auto r2 = Conv2D<float>::conv2d(x, f, b, padding, stride);
        Eigen::Tensor<float, 0> diff = (r2->data->sum() - r->data->sum());

        REQUIRE(t->grad->data()[i] == Approx(diff(0)));

        t->data->data()[i] = oldValue;
    }
}

void testConv(
        const std::shared_ptr<Tensor<float, 4>> &x,
        const std::shared_ptr<Tensor<float, 4>> &f,
        const std::shared_ptr<Tensor<float, 1>> &b,
        int padding,
        int stride) {

    auto r = Conv2D<float>::conv2d(x, f, b, padding, stride);
    auto rReference = conv(*x->data, *f->data, *b->data, padding, stride);

    r->backward(1);

    REQUIRE(tensorEqual<4>(*r->data, rReference));

    // numerical gradient check
    // conv ist just a linear transformation i.e. we can easily compute the numerical gradient
    testGradientNumerically(x, x, f, b, r, padding, stride);
    testGradientNumerically(f, x, f, b, r, padding, stride);
    testGradientNumerically(b, x, f, b, r, padding, stride);
}

TEST_CASE("conv2d") {
    SECTION("odd filter size") {
        auto x = trange<4>({4, 8, 8, 2}, true, 50);
        auto f = trange<4>({4, 3, 3, 8}, true, 50);
        auto b = trange<1>({8});

        SECTION("padding = 0, stride = 1") {
            testConv(x, f, b, 0, 1);
        }
        SECTION("padding = 1, stride = 1") {
            testConv(x, f, b, 1, 1);
        }
        SECTION("padding = 0, stride = 2") {
            testConv(x, f, b, 0, 2);
        }
        SECTION("padding = 1, stride = 2") {
            testConv(x, f, b, 1, 2);
        }
        SECTION("bias = null") {
            auto r = Conv2D<float>::conv2d(x, f, nullptr, 1, 1);
            auto rReference = conv(*x->data, *f->data, b->data->constant(0), 1, 1);

            REQUIRE(tensorEqual<4>(*r->data, rReference));
        }
    }
    SECTION("even filter size") {
        auto x = trange<4>({4, 8, 8, 2}, true, 50);
        auto f = trange<4>({4, 4, 4, 8}, true, 50);
        auto b = trange<1>({8});

        SECTION("padding = 0, stride = 1") {
            testConv(x, f, b, 0, 1);
        }
        SECTION("padding = 1, stride = 1") {
            testConv(x, f, b, 1, 1);
        }
        SECTION("padding = 0, stride = 2") {
            testConv(x, f, b, 0, 2);
        }
        SECTION("padding = 1, stride = 2") {
            testConv(x, f, b, 1, 2);
        }
    }
}