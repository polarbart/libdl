
#include <catch2/catch.hpp>
#include "../../src/ops/Pow.h"
#include "../Helper.h"

TEST_CASE("pow") {
    auto x = trange<2>({16, 8});

    SECTION("positive power") {
        float p = 3.14159;
        auto r = Pow<float, 2>::pow(x, p);
        auto grad = setGradAndBackward<2>(r);

        REQUIRE(tensorEqual<2>(*r->data, x->data->pow(p)));
        REQUIRE(tensorEqual<2>(*x->grad, x->data->constant(p) * x->data->pow(p - 1) * grad));
    }
    SECTION("negative power") {
        float p = -3.14159;
        auto r = Pow<float, 2>::pow(x, p);
        auto grad = setGradAndBackward<2>(r);

        REQUIRE(tensorEqual<2>(*r->data, x->data->pow(p)));
        REQUIRE(tensorEqual<2>(*x->grad, x->data->constant(p) * x->data->pow(p - 1) * grad));
    }

    SECTION("power of 1") {
        float p = 1;
        auto r = Pow<float, 2>::pow(x, p);
        auto grad = setGradAndBackward<2>(r);

        REQUIRE(tensorEqual<2>(*r->data, *x->data));
        REQUIRE(tensorEqual<2>(*x->grad, grad));
    }

    SECTION("power of 0") {
        float p = 0;
        auto r = Pow<float, 2>::pow(x, p);
        auto grad = setGradAndBackward<2>(r);

        REQUIRE(tensorEqual<2>(*r->data, x->data->constant(1)));
        REQUIRE(tensorEqual<2>(*x->grad, x->data->constant(0)));
    }
}