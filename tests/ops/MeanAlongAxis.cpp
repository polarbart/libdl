
#include <catch2/catch.hpp>
#include "../../src/ops/MeanAlongAxes.h"
#include "../Helper.h"

TEST_CASE("mean along axis") {
    auto x = trange<3>({16, 8, 4});

    std::array<int, 2> mean{0, 2};
    auto r = MeanAlongAxes<float, 3, 2>::mean(x, mean);

    auto grad = setGradAndBackward<1>(r);

    REQUIRE(tensorEqual<1>(*r->data, x->data->mean(mean)));

    std::array<long, 3> reshape {1, x->data->dimension(1), 1};
    std::array<long, 3> broadcast {x->data->dimension(0), 1, x->data->dimension(2)};

    int scale = x->data->dimension(0) * x->data->dimension(2);
    REQUIRE(tensorEqual<3>(*x->grad, grad.reshape(reshape).broadcast(broadcast) / x->grad->constant(scale)));
}