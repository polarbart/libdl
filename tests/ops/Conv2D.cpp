
#include <catch2/catch.hpp>
#include "../../src/ops/Conv2D.h"
#include "../Helper.h"

Eigen::Tensor<std::float_t, 4> conv(
        const Eigen::Tensor<std::float_t, 4> &x,
        const Eigen::Tensor<std::float_t, 4> &f,
        const Eigen::Tensor<std::float_t, 1> &b,
        std::int64_t padding,
        std::int64_t stride) {

    std::array<std::int64_t, 4> newDims{
            f.dimension(3),
            (x.dimension(1) - f.dimension(1) + 2 * padding) / stride + 1,
            (x.dimension(2) - f.dimension(2) + 2 * padding) / stride + 1,
            x.dimension(3)
    };

    Eigen::array<Eigen::IndexPair<std::int64_t>, 4> ePadding {
            Eigen::IndexPair<std::int64_t>(0, 0),
            Eigen::IndexPair<std::int64_t>(padding, padding),
            Eigen::IndexPair<std::int64_t>(padding, padding),
            Eigen::IndexPair<std::int64_t>(0, 0)
    };
    Eigen::Tensor<std::float_t, 4> padded = x.pad(ePadding);

    Eigen::Tensor<std::float_t, 4> result(newDims);

    for (std::int64_t n = 0; n < x.dimension(3); n++)
        for (std::int64_t w = 0; w < newDims[2]; w++)
            for (std::int64_t h = 0; h < newDims[1]; h++)
                for (std::int64_t cf = 0; cf < f.dimension(3); cf++) {
                    std::float_t sum = 0;
                    for (std::int64_t cw = 0; cw < f.dimension(2); cw++)
                        for (std::int64_t ch = 0; ch < f.dimension(1); ch++)
                            for (std::int64_t c = 0; c < f.dimension(0); c++)
                                sum += f(c, ch, cw, cf) * padded(c, stride * h + ch, stride * w + cw, n);
                    result(cf, h, w, n) = sum + b(cf);

                }
    return result;

}

template <std::int64_t N>
void testGradientNumerically(
        const std::shared_ptr<Tensor<std::float_t, N>> &t,
        const std::shared_ptr<Tensor<std::float_t, 4>> &x,
        const std::shared_ptr<Tensor<std::float_t, 4>> &f,
        const std::shared_ptr<Tensor<std::float_t, 1>> &b,
        const std::shared_ptr<Tensor<std::float_t, 4>> &r,
        std::int64_t padding,
        std::int64_t stride) {

    for (std::int64_t i = 0; i < t->data->size(); i++) {
        std::float_t oldValue = t->data->data()[i];
        t->data->data()[i] += 1;

        auto r2 = Conv2D<std::float_t>::conv2d(x, f, b, padding, stride);
        Eigen::Tensor<std::float_t, 0> diff = (r2->data->sum() - r->data->sum());

        REQUIRE(t->grad->data()[i] == Approx(diff(0)));

        t->data->data()[i] = oldValue;
    }
}

void testConv(
        const std::shared_ptr<Tensor<std::float_t, 4>> &x,
        const std::shared_ptr<Tensor<std::float_t, 4>> &f,
        const std::shared_ptr<Tensor<std::float_t, 1>> &b,
        std::int64_t padding,
        std::int64_t stride) {

    auto r = Conv2D<std::float_t>::conv2d(x, f, b, padding, stride);
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
            auto r = Conv2D<std::float_t>::conv2d(x, f, nullptr, 1, 1);
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