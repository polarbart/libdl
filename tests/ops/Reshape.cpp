
#include <catch2/catch.hpp>
#include "../../src/ops/Reshape.h"
#include "../Helper.h"

TEST_CASE("reshape") {
    auto x = trange<1>({16*8*4});

    std::array<std::int64_t, 3> reshape {16, 8, 4};
    auto r = Reshape<std::float_t, 1, 3>::reshape(x, reshape);

    auto grad = setGradAndBackward<3>(r);

    REQUIRE(tensorEqual<3>(*r->data, x->data->reshape(reshape)));
    REQUIRE(tensorEqual<1>(*x->grad, grad.reshape(std::array<std::int64_t, 1>{16*8*4})));
}