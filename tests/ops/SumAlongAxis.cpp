
#include <catch2/catch.hpp>
#include "../../src/ops/SumAlongAxes.h"
#include "../Helper.h"

TEST_CASE("sum along axis") {
    auto x = trange<3>({16, 8, 4});

    std::array<std::int64_t, 2> sum {0, 2};
    auto r = SumAlongAxes<std::float_t, 3, 2>::sum(x, sum);

    auto grad = setGradAndBackward<1>(r);

    REQUIRE(tensorEqual<1>(*r->data, x->data->sum(sum)));

    std::array<std::int64_t, 3> reshape {1, x->data->dimension(1), 1};
    std::array<std::int64_t, 3> broadcast {x->data->dimension(0), 1, x->data->dimension(2)};

    REQUIRE(tensorEqual<3>(*x->grad, grad.reshape(reshape).broadcast(broadcast)));
}