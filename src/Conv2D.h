#ifndef LIBDL_CONV2D_H
#define LIBDL_CONV2D_H

#include <iostream>
#include "CNode.h"
#include "Utils.h"

#define R 4

template <typename D>
class Conv2D : public CNode<D, R> {
public:
    Conv2D(const std::shared_ptr<Tensor<D, R>> &a,
           const std::shared_ptr<Tensor<D, R>> &filter,
           const std::shared_ptr<Tensor<D, 1>> &bias,
           const std::shared_ptr<Tensor<D, R>> &result,
           int padding)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, filter->gradFn, bias->gradFn}), result),
            a(a->eTensor),
            filter(filter->eTensor),
            ca(a->gradFn),
            cfilter(filter->gradFn),
            cbias(bias->gradFn),
            padding(padding) {}

    Conv2D(const std::shared_ptr<Tensor<D, R>> &a,
           const std::shared_ptr<Tensor<D, R>> &filter,
           const std::shared_ptr<Tensor<D, R>> &result,
           int padding)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, filter->gradFn}), result),
              a(a->eTensor),
              filter(filter->eTensor),
              ca(a->gradFn),
              cfilter(filter->gradFn),
              cbias(std::nullopt),
              padding(padding) {}
    /*
     * a (h, w, c, n)
     * f (a, b, c, c')
     * b (c')
     * result (h', w', c', n)
     * */
    static std::shared_ptr<Tensor<D, R>> conv2d(const std::shared_ptr<Tensor<D, R>> &a, const std::shared_ptr<Tensor<D, R>> &filter, const std::shared_ptr<Tensor<D, 1>> &bias, int padding) {
        std::array<long, 4> newDims {
                (a->eTensor->dimension(0) - filter->eTensor->dimension(0) + 1 + 2 * padding),
                (a->eTensor->dimension(1) - filter->eTensor->dimension(1) + 1 + 2 * padding),
                filter->eTensor->dimension(3),
                a->eTensor->dimension(3),
        };
        auto result = std::make_shared<Tensor<D, R>>(newDims);

        Eigen::array<Eigen::IndexPair<int>, R> ePadding {
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(padding, padding),
            Eigen::IndexPair<int>(0, 0),
            Eigen::IndexPair<int>(0, 0)
        };
        Eigen::array<ptrdiff_t, 3> convDims {0, 1, 2};

        auto intermediate = a->eTensor->pad(ePadding);

        for (long i = 0; i < filter->eTensor->dimension(3); i++) {
            auto intermediate2 = intermediate.convolve(filter->eTensor->chip(i, 3), convDims).chip(0, 2);
            if (bias != nullptr)
                result->eTensor->chip(i, 2) = intermediate2 + intermediate2.constant((*bias->eTensor)(i));
            else
                result->eTensor->chip(i, 2) = intermediate2;
        }

        if (a->needsGradient() || filter->needsGradient() || (bias != nullptr && bias->needsGradient())) {
            if (bias != nullptr)
                result->setGradFn(std::make_shared<Conv2D<D>>(a, filter, bias, result, padding));
            else
                result->setGradFn(std::make_shared<Conv2D<D>>(a, filter, result, padding));
        }

        return result;
    }

    void computeGradients() override {
        if (ca.has_value()) {
            Eigen::array<ptrdiff_t, 3> convDims {0, 1, 2};
            int backPadding = (ca.value()->shape[0] - CNode<D, R>::shape[0] + filter->dimension(0) - 1) / 2;
            Eigen::array<Eigen::IndexPair<int>, R> ePadding {
                    Eigen::IndexPair<int>(backPadding, backPadding),
                    Eigen::IndexPair<int>(backPadding, backPadding),
                    Eigen::IndexPair<int>(0, 0),
                    Eigen::IndexPair<int>(0, 0)
            };
            auto intermediate = CNode<D, R>::grad->pad(ePadding);
            ca.value()->emptyGrad();
            for (long i = 0; i < filter->dimension(2); i++)
                ca.value()->grad->chip(i, 2) = intermediate.convolve(filter->chip(i, 2).reverse(Eigen::array<bool, 3> {true, true, false}), convDims).chip(0, 2);
        }
        if (cfilter.has_value()) {
            Eigen::array<ptrdiff_t, 3> convDims {0, 1, 2};
            Eigen::array<Eigen::IndexPair<int>, R> ePadding {
                    Eigen::IndexPair<int>(padding, padding),
                    Eigen::IndexPair<int>(padding, padding),
                    Eigen::IndexPair<int>(0, 0),
                    Eigen::IndexPair<int>(0, 0)
            };

            Eigen::array<long, 4> offsets {0, 0, -1, 0};
            Eigen::array<long, 4> extents {
                    CNode<D, R>::grad->dimension(0),
                    CNode<D, R>::grad->dimension(1),
                    1,
                    CNode<D, R>::grad->dimension(3)
            };
            Eigen::array<int, 1> sumDim {3};
            auto intermediate = a->pad(ePadding);
            cfilter.value()->emptyGrad();
            for (long i = 0; i < filter->dimension(3); i++) {
                offsets[2] = i;
                cfilter.value()->grad->chip(i, 3) = intermediate.convolve(CNode<D, R>::grad->slice(offsets, extents).sum(sumDim), convDims).mean(sumDim);
            }
        }
        if (cbias.has_value()) {
            cbias.value()->addGrad(CNode<D, R>::grad->sum(Eigen::array<int, 3> {0, 1, 3}));
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> a;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> filter;
    std::optional<std::shared_ptr<CNode<D, R>>> ca;
    std::optional<std::shared_ptr<CNode<D, R>>> cfilter;
    std::optional<std::shared_ptr<CNode<D, 1>>> cbias;
    int padding;
};

#undef R

#endif //LIBDL_CONV2D_H
