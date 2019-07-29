
#include <catch2/catch.hpp>
#include "../../src/ops/Sum.h"
#include "../Helper.h"

TEST_CASE("sum") {
    auto x = trange<3>({16, 8, 4});
    auto r = Sum<float, 3>::sum(x);
    r->backward(5);
    float size = x->data->size();
    // Little Gauss formula
    REQUIRE((*r->data)(0) == (size*(size+1))/2);
    REQUIRE(tensorEqual<3>(*x->grad, x->grad->constant(5)));
}
