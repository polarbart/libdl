
#include <catch2/catch.hpp>
#include "../../src/ops/Relu.h"
#include "../Helper.h"

TEST_CASE("ReLU") {
    auto x = random<2>({16, 8});
    auto r = Relu<std::float_t, 2>::relu(x);

    auto grad = setGradAndBackward<2>(r);

    for (std::int64_t i = 0; i < x->data->dimension(1); i++)
        for (std::int64_t j = 0; j < x->data->dimension(0); j++)
            if ((*x->data)(j, i) >= 0) {
                REQUIRE((*r->data)(j, i) == (*x->data)(j, i));
                REQUIRE((*x->grad)(j, i) == grad(j, i));
            } else {
                REQUIRE((*r->data)(j, i) == 0);
                REQUIRE((*x->grad)(j, i) == 0);
            }
}