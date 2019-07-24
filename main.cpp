
//#define CATCH_CONFIG_MAIN
//#include <catch2/catch.hpp>

#include "src/Tensor.h"
#include "src/ops/Add.h"

/*
TEST_CASE("add") {
    auto t = std::make_shared<Tensor<float, 2>>(std::array<long, 2> {2, 2});
    REQUIRE(2*2==4);
}
*/

int main() {

    auto t = std::make_shared<Tensor<float, 2>>(std::array<long, 2> {2, 2});
    std::cout << *t->eTensor << std::endl;
    std::cout << 8743 << std::endl;
}
