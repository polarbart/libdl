
#include <catch2/catch.hpp>
#include "../../src/ops/Linear.h"
#include "../Helper.h"

TEST_CASE("linear layer") {
    auto x = trange<2>({16, 8});
    auto w = random<2>({16, 32});
    auto b = trange<1>({32});

    auto r = Linear<std::float_t>::linear(w, x, b); // 32 x 8

    auto grad = setGradAndBackward<2>(r);

    for (std::int64_t i = 0; i < r->data->dimension(1); i++)
        for (std::int64_t j = 0; j < r->data->dimension(0); j++) {
            std::float_t forward = 0;
            for (std::int64_t k = 0; k < x->data->dimension(0); k++)
                forward += (*x->data)(k, i) * (*w->data)(k, j);
            REQUIRE((*r->data)(j, i) == forward + (*b->data)(j));
        }

    // gradient for x
    for (std::int64_t i = 0; i < x->data->dimension(1); i++)
        for (std::int64_t j = 0; j < x->data->dimension(0); j++) {
            std::float_t backward = 0;
            for (std::int64_t k = 0; k < grad.dimension(0); k++)
                backward += grad(k, i) * (*w->data)(j, k);
            REQUIRE((*x->grad)(j, i) == backward);
        }

    // gradient for w
    for (std::int64_t i = 0; i < w->data->dimension(1); i++)
        for (std::int64_t j = 0; j < w->data->dimension(0); j++) {
            std::float_t backward = 0;
            for (std::int64_t k = 0; k < grad.dimension(1); k++)
                backward += grad(i, k) * (*x->data)(j, k);
            REQUIRE((*w->grad)(j, i) == backward);
        }

    // gradient for b
    REQUIRE(tensorEqual<1>(*b->grad, grad.sum(std::array<std::int64_t, 1> {1})));
}