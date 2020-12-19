
#include <catch2/catch.hpp>
#include "../../src/ops/MeanAlongAxes.h"
#include "../Helper.h"

TEST_CASE("mean along axis") {
    auto x = trange<3>({16, 8, 4});

    std::array<std::int64_t, 2> mean{0, 2};
    auto r = MeanAlongAxes<std::float_t, 3, 2>::mean(x, mean);

    auto grad = setGradAndBackward<1>(r);

    REQUIRE(tensorEqual<1>(*r->data, x->data->mean(mean)));

    std::array<std::int64_t, 3> reshape {1, x->data->dimension(1), 1};
    std::array<std::int64_t, 3> broadcast {x->data->dimension(0), 1, x->data->dimension(2)};

    std::int64_t scale = x->data->dimension(0) * x->data->dimension(2);
    REQUIRE(tensorEqual<3>(*x->grad, grad.reshape(reshape).broadcast(broadcast) / x->grad->constant(static_cast<std::float_t>(scale))));
}