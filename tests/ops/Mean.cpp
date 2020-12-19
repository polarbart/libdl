
#include <catch2/catch.hpp>
#include "../../src/ops/Mean.h"
#include "../Helper.h"

TEST_CASE("mean") {
    auto x = trange<3>({16, 8, 4});
    auto r = Mean<std::float_t, 3>::mean(x);
    r->backward(5);
    auto size = static_cast<std::float_t>(x->data->size());
    // Little Gauss formula
    REQUIRE((*r->data)(0) == (size*(size+1))/2/size);
    REQUIRE(tensorEqual<3>(*x->grad, x->grad->constant(5.f / size)));
}
