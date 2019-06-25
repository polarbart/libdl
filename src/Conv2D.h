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

        std::array<long, R> newDims {
                (a->eTensor->dimension(0) - filter->eTensor->dimension(0) + 1 + 2 * padding),
                (a->eTensor->dimension(1) - filter->eTensor->dimension(1) + 1 + 2 * padding),
                filter->eTensor->dimension(3),
                a->eTensor->dimension(3),
        };


        auto r = myConvolution(*a->eTensor, *filter->eTensor, padding);

        std::shared_ptr<Tensor<D, R>> result;
        if (bias != nullptr) {
            Eigen::array<long, R> reshape {1, 1, bias->eTensor->dimension(0), 1};
            Eigen::array<long, R> broadcast {newDims[0], newDims[1], 1, newDims[3]};
            result = std::make_shared<Tensor<D, R>>(r + bias->eTensor->reshape(reshape).broadcast(broadcast), newDims);
        } else
            result = std::make_shared<Tensor<D, R>>(r, newDims);

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
            int backPadding = (ca.value()->shape[0] - CNode<D, R>::shape[0] + filter->dimension(0) - 1) / 2;
            Eigen::Tensor<D, R> backFilter = filter->reverse(Eigen::array<bool, 4> {true, true, false, false}).shuffle(Eigen::array<int, 4> {0, 1, 3, 2});
            ca.value()->addGrad(myConvolution(*CNode<D, R>::grad, backFilter, backPadding));
        }
        if (cfilter.has_value()) {
            Eigen::array<Eigen::IndexPair<int>, R> ePadding{
                    Eigen::IndexPair<int>(padding, padding),
                    Eigen::IndexPair<int>(padding, padding),
                    Eigen::IndexPair<int>(0, 0),
                    Eigen::IndexPair<int>(0, 0),
            };
            auto padded = a->pad(ePadding).eval();

            Eigen::array<ptrdiff_t, R> patchDims {CNode<D, R>::grad->dimension(0), CNode<D, R>::grad->dimension(1), 1, a->dimension(3)};
            auto patches = padded.extract_patches(patchDims);

            Eigen::array<long, R-1> reshape {patchDims[0]*patchDims[1], patchDims[3], filter->dimension(0) * filter->dimension(1) * a->dimension(2)};
            auto im2col = patches.reshape(reshape);
            auto i2cFilter = CNode<D, R>::grad->sum(Eigen::array<int, 1> {3}).reshape(Eigen::array<long, 2> {reshape[0], CNode<D, R>::grad->dimension(2)});
            auto conv = i2cFilter.contract(im2col, Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(0, 0)});

            Eigen::array<int, R-1> shuffle {2, 0, 1};
            Eigen::array<long, R+1> reshape2 {filter->dimension(0), filter->dimension(1), a->dimension(2), CNode<D, R>::grad->dimension(2), a->dimension(3)};
            cfilter.value()->addGrad(conv.shuffle(shuffle).eval().reshape(reshape2).mean(Eigen::array<int, 1> {4}));
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

    /*
     * For an awkward reason my implementation of conv is 30 times faster than the Eigen implementation
     * */
    template <typename OtherDerived, typename OtherDerived2>
    static auto myConvolution(const OtherDerived &a, const OtherDerived2 &filter, int padding) {
        Eigen::array<Eigen::IndexPair<int>, R> ePadding{
                Eigen::IndexPair<int>(padding, padding),
                Eigen::IndexPair<int>(padding, padding),
                Eigen::IndexPair<int>(0, 0),
                Eigen::IndexPair<int>(0, 0)
        };
        //auto padded = a.pad(ePadding);
        auto padded = a.pad(ePadding).eval();

        Eigen::array<ptrdiff_t, R> patchDims{filter.dimension(0), filter.dimension(1), a.dimension(2), a.dimension(3)};
        auto patches = padded.extract_patches(patchDims);

        int newHeight = a.dimension(0) - filter.dimension(0) + 2 * padding + 1;
        int newWidth = a.dimension(1) - filter.dimension(1) + 2 * padding + 1;
        Eigen::array<long, R - 1> reshape{patchDims[0] * patchDims[1] * patchDims[2], patchDims[3],
                                          newHeight * newWidth};
        auto im2col = patches.reshape(reshape);
        auto i2cFilter = filter.reshape(Eigen::array<long, 2>{reshape[0], filter.dimension(3)});
        auto conv = i2cFilter.contract(im2col, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});

        Eigen::array<int, R - 1> shuffle{2, 0, 1};
        Eigen::array<long, R> reshape2{newHeight, newWidth, filter.dimension(3), a.dimension(3)};
        return conv.shuffle(shuffle).eval().reshape(reshape2);
    }
};

#undef R

#endif //LIBDL_CONV2D_H


/*
static std::shared_ptr<Tensor<D, R>> conv2d(const std::shared_ptr<Tensor<D, R>> &a, const std::shared_ptr<Tensor<D, R>> &filter, const std::shared_ptr<Tensor<D, 1>> &bias, int padding) {
        std::array<long, 4> newDims{
                (a->eTensor->dimension(0) - filter->eTensor->dimension(0) + 1 + 2 * padding),
                (a->eTensor->dimension(1) - filter->eTensor->dimension(1) + 1 + 2 * padding),
                filter->eTensor->dimension(3),
                a->eTensor->dimension(3),
        };
        auto result = std::make_shared<Tensor<D, R>>(newDims);

        Eigen::array<Eigen::IndexPair<int>, R> ePadding{
                Eigen::IndexPair<int>(padding, padding),
                Eigen::IndexPair<int>(padding, padding),
                Eigen::IndexPair<int>(0, 0),
                Eigen::IndexPair<int>(0, 0)
        };
        Eigen::array<ptrdiff_t, 3> convDims{0, 1, 2};

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
    template <typename OtherDerived>
    static auto myConvolution(const OtherDerived &a, const Eigen::TensorBase<Eigen::Tensor<D, R>> &filter) {
        Eigen::array<ptrdiff_t, R> patchDims {filter.dimension(0), filter.dimension(1), a.dimension(2), a.dimension(3)};
        auto patches = a.extract_patches(patchDims);
    }







        void computeGradients() override {
        if (ca.has_value()) {
            int backPadding = (ca.value()->shape[0] - CNode<D, R>::shape[0] + filter->dimension(0) - 1) / 2;
            Eigen::Tensor<D, R> backFilter = filter->reverse(Eigen::array<bool, 4> {true, true, false, false}).shuffle(Eigen::array<int, 4> {0, 1, 3, 2});
            ca.value()->addGrad(myConvolution(*CNode<D, R>::grad, backFilter, backPadding));
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










                Eigen::array<Eigen::IndexPair<int>, R> ePadding{
                    Eigen::IndexPair<int>(padding, padding),
                    Eigen::IndexPair<int>(padding, padding),
                    Eigen::IndexPair<int>(0, 0),
                    Eigen::IndexPair<int>(0, 0),
            };
            auto padded = a->pad(ePadding);

            Eigen::array<ptrdiff_t, R> patchDims {CNode<D, R>::grad->dimension(0), CNode<D, R>::grad->dimension(1), 1, a->dimension(3)};
            auto patches = padded.extract_patches(patchDims);

            Eigen::array<long, R-1> reshape {patchDims[0]*patchDims[1], patchDims[3], filter->dimension(0) * filter->dimension(1) * a->dimension(2)};
            auto im2col = patches.reshape(reshape);
            auto i2cFilter = CNode<D, R>::grad->sum(Eigen::array<int, 1> {3}).reshape(Eigen::array<long, 2> {reshape[0], CNode<D, R>::grad->dimension(2)});
            auto conv = i2cFilter.contract(im2col, Eigen::array<Eigen::IndexPair<int>, 1> {Eigen::IndexPair<int>(0, 0)});

            Eigen::array<int, R-1> shuffle {2, 0, 1};
            Eigen::array<long, R+1> reshape2 {filter->dimension(0), filter->dimension(1), a->dimension(2), CNode<D, R>::grad->dimension(2), a->dimension(3)};
            cfilter.value()->addGrad(conv.shuffle(shuffle).reshape(reshape2).sum(Eigen::array<int, 1> {3}));

 */