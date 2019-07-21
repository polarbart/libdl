
#include "test_helper.h"

TEST_CASE("add") {
    SECTION("simple addition") {
        auto a = make_tensor({16, 8});
        auto b = make_tensor({16, 8});
        auto c = Add<float, 2, 2>::add(a, b);
        for (int i = 0; i < a->eTensor->dimension(1); i++)
            for (int j = 0; j < a->eTensor->dimension(0); j++)
                REQUIRE(((*a->eTensor)(j, i) + (*b->eTensor)(j, i)) == (*c->eTensor)(j, i));
    }
    SECTION("broadcasted addition") {
        auto a = make_tensor({16, 8, 4});
        auto b = make_tensor({16, 8});
        SECTION("ab") {
            auto c = Add<float, 3, 2>::add(a, b);
            for (int i = 0; i < a->eTensor->dimension(2); i++)
                for (int j = 0; j < a->eTensor->dimension(1); j++)
                    for (int k = 0; k < a->eTensor->dimension(0); k++)
                        REQUIRE(((*a->eTensor)(k, j, i) + (*b->eTensor)(k, j)) == (*c->eTensor)(k, j, i));
        }
        SECTION("ba") {
            auto c = Add<float, 2, 3>::add(b, a);
            for (int i = 0; i < a->eTensor->dimension(2); i++)
                for (int j = 0; j < a->eTensor->dimension(1); j++)
                    for (int k = 0; k < a->eTensor->dimension(0); k++)
                        REQUIRE(((*a->eTensor)(k, j, i) + (*b->eTensor)(k, j)) == (*c->eTensor)(k, j, i));
        }

    }
}