
#include <catch2/catch.hpp>
#include "../../src/ops/Sigmoid.h"
#include "../Helper.h"

TEST_CASE("sigmoid") {
    auto x = trange<2>({16, 8});

    auto r = Sigmoid<float, 2>::sigmoid(x);

    auto grad = setGradAndBackward<2>(r);

    REQUIRE(tensorEqual<2>(*r->data, x->data->sigmoid()));

    REQUIRE(tensorEqual<2>(*x->grad, *r->data * (r->data->constant(1) - *r->data) * grad));
}