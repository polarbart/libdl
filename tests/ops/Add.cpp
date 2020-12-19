
#include <catch2/catch.hpp>
#include "../../src/ops/Add.h"
#include "../Helper.h"

TEST_CASE("add") {
    SECTION("simple addition") {
        auto a = trange<2>({16, 8});
        auto b = trange<2>({16, 8});
        auto c = Add<std::float_t, 2, 2>::add(a, b);

        auto grad = setGradAndBackward<2>(c);

        REQUIRE(tensorEqual<2>(*c->data, *a->data + *b->data));
        REQUIRE(tensorEqual<2>(*a->grad, grad));
        REQUIRE(tensorEqual<2>(*b->grad, grad));
    }
    SECTION("broadcasted addition") {
        auto a = trange<3>({16, 8, 4});
        auto b = trange<2>({16, 8});
        std::array<std::int64_t, 3> reshape {16, 8, 1};
        std::array<std::int64_t, 3> broadcast {1, 1, 4};

        SECTION("ab") {
            auto c = Add<std::float_t, 3, 2>::add(a, b);

            auto grad = setGradAndBackward<3>(c);

            REQUIRE(tensorEqual<3>(*c->data, *a->data + b->data->reshape(reshape).broadcast(broadcast)));
            REQUIRE(tensorEqual<3>(*a->grad, grad));
            REQUIRE(tensorEqual<2>(*b->grad, grad.sum(std::array<std::int64_t, 1> {2})));
        }
        SECTION("ba") {
            auto c = Add<std::float_t, 2, 3>::add(b, a);

            auto grad = setGradAndBackward<3>(c);

            REQUIRE(tensorEqual<3>(*c->data, *a->data + b->data->reshape(reshape).broadcast(broadcast)));
            REQUIRE(tensorEqual<3>(*a->grad, grad));
            REQUIRE(tensorEqual<2>(*b->grad, grad.sum(std::array<std::int64_t, 1> {2})));
        }
    }
}