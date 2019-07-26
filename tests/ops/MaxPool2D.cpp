
#include <catch2/catch.hpp>
#include "../../src/ops/MaxPool2D.h"
#include "../Helper.h"


TEST_CASE("maxpool2d") {
    auto a = random({3, 16, 16, 4});
    auto r = MaxPool2D<float>::maxpool2d(a, 3); // 3x5x5x4
    auto grad = setGradAndBackward<4>(r);

    REQUIRE(r->data->dimension(1) == a->data->dimension(1) / 3);
    REQUIRE(r->data->dimension(2) == a->data->dimension(2) / 3);

    for (int n = 0; n < r->data->dimension(3); n++)
        for (int c = 0; c < r->data->dimension(0); c++)
            for (int w = 0; w < r->data->dimension(2); w++)
                for (int h = 0; h < r->data->dimension(1); h++) {
                    int ah = 0;
                    int aw = 0;
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            if ((*a->data)(c, 3 * h + j, 3 * w + i, n) > (*a->data)(c, 3 * h + ah, 3 * w + aw, n)) {
                                ah = j;
                                aw = i;
                            }

                    REQUIRE((*r->data)(c, h, w, n) == (*a->data)(c, 3 * h + ah, 3 * w + aw, n));

                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            if (j == ah && i == aw)
                                REQUIRE((*a->grad)(c, 3 * h + j, 3 * w + i, n) == grad(c, h, w, n));
                            else
                                REQUIRE((*a->grad)(c, 3 * h + j, 3 * w + i, n) == 0);

                }


}