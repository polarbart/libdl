
#include <catch2/catch.hpp>
#include "../../src/ops/Sub.h"
#include "../Helper.h"

TEST_CASE("sub") {
    SECTION("simple substraction") {
        auto a = trange<2>({16, 8});
        auto b = trange<2>({16, 8});
        auto c = Sub<float, 2, 2>::sub(a, b);

        auto grad = setGradAndBackward<2>(c);

        REQUIRE(tensorEqual<2>(*c->data, *a->data - *b->data));
        REQUIRE(tensorEqual<2>(*a->grad, grad));
        REQUIRE(tensorEqual<2>(*b->grad, -grad));
    }
    SECTION("broadcasted substraction") {
        auto a = trange<3>({16, 8, 4});
        auto b = trange<2>({16, 8});
        std::array<long, 3> reshape {16, 8, 1};
        std::array<long, 3> broadcast {1, 1, 4};

        SECTION("ab") {
            auto c = Sub<float, 3, 2>::sub(a, b);

            auto grad = setGradAndBackward<3>(c);

            REQUIRE(tensorEqual<3>(*c->data, *a->data - b->data->reshape(reshape).broadcast(broadcast)));
            REQUIRE(tensorEqual<3>(*a->grad, grad));
            REQUIRE(tensorEqual<2>(*b->grad, -grad.sum(std::array<long, 1> {2})));
        }
        SECTION("ba") {
            auto c = Sub<float, 2, 3>::sub(b, a);

            auto grad = setGradAndBackward<3>(c);

            REQUIRE(tensorEqual<3>(*c->data, b->data->reshape(reshape).broadcast(broadcast) - *a->data));
            REQUIRE(tensorEqual<3>(*a->grad, -grad));
            REQUIRE(tensorEqual<2>(*b->grad, grad.sum(std::array<long, 1> {2})));
        }
    }
}