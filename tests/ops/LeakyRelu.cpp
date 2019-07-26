
#include <catch2/catch.hpp>
#include "../../src/ops/LeakyRelu.h"
#include "../Helper.h"

TEST_CASE("leaky ReLU") {
    auto x = random({16, 8});
    auto r = LeakyRelu<float, 2>::leakyRelu(x, 0.01);

    auto grad = setGradAndBackward<2>(r);

    for (int i = 0; i < x->data->dimension(1); i++)
        for (int j = 0; j < x->data->dimension(0); j++)
            if ((*x->data)(j, i) >= 0) {
                REQUIRE((*r->data)(j, i) == (*x->data)(j, i));
                REQUIRE((*x->grad)(j, i) == grad(j, i));
            } else {
                REQUIRE((*r->data)(j, i) == (*x->data)(j, i) * 0.01f);
                REQUIRE((*x->grad)(j, i) == grad(j, i) * 0.01f);
            }
}