
#include <catch2/catch.hpp>
#include "../../src/ops/MatMul.h"
#include "../Helper.h"

TEST_CASE("matmul") {
    auto a = trange<2>({8, 16});
    auto b = trange<3>({16, 4, 2});

    auto r = MatMul<std::float_t, 2, 3>::matmul(a, b); // 8x4x2

    auto grad = setGradAndBackward<3>(r);

    auto rReference = a->data->contract(*b->data, std::array<Eigen::IndexPair<std::int64_t>, 1> {Eigen::IndexPair<std::int64_t> (1, 0)});
    REQUIRE(tensorEqual<3>(*r->data, rReference));

    auto aGradReference = grad.contract(*b->data, std::array<Eigen::IndexPair<std::int64_t>, 2> {Eigen::IndexPair<std::int64_t> (1, 1), Eigen::IndexPair<std::int64_t> (2, 2)});
    REQUIRE(tensorEqual<2>(*a->grad, aGradReference));

    auto bGradReference = a->data->contract(grad, std::array<Eigen::IndexPair<std::int64_t>, 1> {Eigen::IndexPair<std::int64_t> (0, 0)});
    REQUIRE(tensorEqual<3>(*b->grad, bGradReference));

}