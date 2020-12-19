
#include <catch2/catch.hpp>
#include "../../src/ops/CrossEntropyWithLogits.h"
#include "../Helper.h"

TEST_CASE("cross entropy with logits") {
    auto x = trange<2>({16, 32});
    auto y = constant<2>({16, 32}, 0, false);

    // one hot encoding
    for (std::int64_t i = 0; i < y->data->dimension(1); i++)
        (*y->data)(i % y->data->dimension(0), i) = 1;

    auto ce = CrossEntropyWithLogits<std::float_t>::crossEntropyWithLogits(x, y);
    ce->backward();

    SECTION("normal test") {
        std::array<std::int64_t, 2> reshape{1, x->data->dimension(1)};
        std::array<std::int64_t, 2> broadcast{x->data->dimension(0), 1};

        auto intermediate = (*x->data - x->data->maximum(Eigen::array<std::int64_t, 1>{0}).eval().reshape(reshape).broadcast(
                broadcast)).exp();
        Eigen::Tensor<std::float_t, 2> softmax =
                intermediate / intermediate.sum(Eigen::array<std::int64_t, 1>{0}).eval().reshape(reshape).broadcast(broadcast);
        Eigen::Tensor<std::float_t, 0> ceReference = (-softmax.log() * *y->data).mean();

        REQUIRE((*ce->data)(0) == Approx(ceReference(0) * y->data->dimension(0)));
        REQUIRE(tensorEqual<2>(*x->grad, (softmax - *y->data) / softmax.constant(static_cast<std::float_t>(y->data->dimension(1)))));
    }

    WHEN("negative gradient is followed") {
        *x->data -= *x->grad;
        auto ce2 = CrossEntropyWithLogits<std::float_t>::crossEntropyWithLogits(x, y);
        THEN("cross entropy should get smaller") {
            REQUIRE((*ce2->data)(0) < (*ce->data)(0));
        }
    }
    WHEN("gradient is followed") {
        *x->data += *x->grad;
        auto ce2 = CrossEntropyWithLogits<std::float_t>::crossEntropyWithLogits(x, y);
        THEN("cross entropy should get larger") {
            REQUIRE((*ce2->data)(0) > (*ce->data)(0));
        }
    }

    WHEN("value for correct class decreases") {
        (*x->data)(0, 0) -= 1;
        auto ce2 = CrossEntropyWithLogits<std::float_t>::crossEntropyWithLogits(x, y);
        THEN("cross entropy should get increase") {
            REQUIRE((*ce2->data)(0) > (*ce->data)(0));
        }
    }
    WHEN("value for correct class increases") {
        (*x->data)(0, 0) += 1;
        auto ce2 = CrossEntropyWithLogits<std::float_t>::crossEntropyWithLogits(x, y);
        THEN("cross entropy should get decrease") {
            REQUIRE((*ce2->data)(0) < (*ce->data)(0));
        }
    }
}