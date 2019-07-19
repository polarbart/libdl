#ifndef LIBDL_CONV2D_H
#define LIBDL_CONV2D_H

#include <iostream>
#include "CNode.h"
#include "Utils.h"

#define R 4
#define DIV_CEIL(x, y) ((x)/(y) + ((x) % (y) != 0))

template <typename D>
class Conv2D : public CNode<D, R> {
public:

    Conv2D(const std::shared_ptr<Tensor<D, R>> &a,
           const std::shared_ptr<Tensor<D, R>> &filter,
           const std::shared_ptr<Tensor<D, 1>> &bias,
           const std::shared_ptr<Tensor<D, R>> &result,
           int padding,
           int stride)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, filter->gradFn, bias->gradFn}), result),
            a(a->eTensor),
            filter(filter->eTensor),
            ca(a->gradFn),
            cfilter(filter->gradFn),
            cbias(bias->gradFn),
            padding(padding),
            stride(stride) {}

    Conv2D(const std::shared_ptr<Tensor<D, R>> &a,
           const std::shared_ptr<Tensor<D, R>> &filter,
           const std::shared_ptr<Tensor<D, R>> &result,
           int padding,
           int stride)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({a->gradFn, filter->gradFn}), result),
              a(a->eTensor),
              filter(filter->eTensor),
              ca(a->gradFn),
              cfilter(filter->gradFn),
              cbias(std::nullopt),
              padding(padding),
              stride(stride) {}
    /*
     * a (h, w, c, n)
     * f (c, a, b, c')
     * b (c')
     * result (h', w', c', n)
     * */
    static std::shared_ptr<Tensor<D, R>> conv2d(const std::shared_ptr<Tensor<D, R>> &a, const std::shared_ptr<Tensor<D, R>> &filter, const std::shared_ptr<Tensor<D, 1>> &bias, int padding, int stride) {

        std::array<long, R> newDims {
                filter->eTensor->dimension(3),
                (a->eTensor->dimension(1) - filter->eTensor->dimension(1) + 2 * padding) / stride + 1,
                (a->eTensor->dimension(2) - filter->eTensor->dimension(2) + 2 * padding) / stride + 1,
                a->eTensor->dimension(3),
        };

        // TODO: implement conv with extract_image_patches(padding_same) and extract_volume_patches

        auto r = myConvolution(*a->eTensor, *filter->eTensor, padding, stride);

        std::shared_ptr<Tensor<D, R>> result;
        if (bias != nullptr) {
            Eigen::array<long, R> reshape {bias->eTensor->dimension(0), 1, 1, 1};
            Eigen::array<long, R> broadcast {1, newDims[1], newDims[2], newDims[3]};
            result = std::make_shared<Tensor<D, R>>(r + bias->eTensor->reshape(reshape).broadcast(broadcast), newDims);
        } else
            result = std::make_shared<Tensor<D, R>>(r, newDims);

        if (a->needsGradient() || filter->needsGradient() || (bias != nullptr && bias->needsGradient())) {
            if (bias != nullptr)
                result->setGradFn(std::make_shared<Conv2D<D>>(a, filter, bias, result, padding, stride));
            else
                result->setGradFn(std::make_shared<Conv2D<D>>(a, filter, result, padding, stride));
        }
        return result;
    }

    void computeGradients() override {
        static Eigen::ThreadPool pool(8);
        static Eigen::ThreadPoolDevice myDevice(&pool, 8);

        if (ca.has_value()) {
            int temp = (ca.value()->shape[1] + filter->dimension(1) - stride * (CNode<D, R>::grad->dimension(1) - 1) - 2);
            int backPadding = temp / 2;
            int additionalBRPadding = temp % 2;
            Eigen::Tensor<D, R> backFilter(filter->dimension(3), filter->dimension(1), filter->dimension(2), filter->dimension(0));
            backFilter.device(myDevice) = filter->reverse(Eigen::array<bool, R> {false, true, true, false}).shuffle(Eigen::array<int, 4> {3, 1, 2, 0}); // TODO parallel
            ca.value()->addGrad(myConvolution(*CNode<D, R>::grad, backFilter, backPadding, 1, stride, additionalBRPadding));
        }
        if (cfilter.has_value()) {
            Eigen::array<long, R + 1> reshape {1,  a->dimension(0), a->dimension(1), a->dimension(2), a->dimension(3)};
            Eigen::array<long, R - 1> reshape2 {a->dimension(1) * a->dimension(2), a->dimension(0) * filter->dimension(1) * filter->dimension(2), a->dimension(3)};
            Eigen::array<long, R + 1> reshape3 {filter->dimension(0), filter->dimension(1), filter->dimension(2), a->dimension(3), filter->dimension(3)};


            auto avol = a->reshape(reshape).extract_volume_patches(1,  a->dimension(1), a->dimension(2), 1, 1, 1, 1, 1, 1, 0, 0, padding, padding, padding, padding);

            auto im2col = avol.reshape(reshape2);

            if (stride == 1) {
                auto i2cFilter = CNode<D, R>::grad->sum(Eigen::array<int, 1>{3}).reshape(Eigen::array<long, 2>{CNode<D, R>::grad->dimension(0), reshape2[0]}).eval();
                auto conv = im2col.contract(i2cFilter, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 1)});
                cfilter.value()->addGrad(conv.reshape(reshape3).mean(Eigen::array<int, 1>{3}));
            } else {
                int dheight = a->dimension(1) - filter->dimension(1) + 2 * padding + 1;
                int dwidth = a->dimension(2) - filter->dimension(2) + 2 * padding + 1;
                Eigen::Tensor<D, R - 1> dialated(CNode<D, R>::grad->dimension(0), dheight, dwidth);
                Eigen::Tensor<D, R - 1> summedGrad(CNode<D, R>::grad->dimension(0), CNode<D, R>::grad->dimension(1), CNode<D, R>::grad->dimension(2));
                summedGrad.device(myDevice) = CNode<D, R>::grad->sum(Eigen::array<int, 1>{3});
                myDialationForFilter(dialated, summedGrad, stride);
                auto i2cFilter = dialated.reshape(Eigen::array<long, 2>{CNode<D, R>::grad->dimension(0), reshape2[0]});
                auto conv = im2col.contract(i2cFilter, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 1)});
                cfilter.value()->addGrad(conv.reshape(reshape3).mean(Eigen::array<int, 1>{3}));
            }
        }
        if (cbias.has_value()) {
            cbias.value()->addGrad(CNode<D, R>::grad->sum(Eigen::array<int, 3> {1, 2, 3}));
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> a;
    std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<D, R>>> filter;
    std::optional<std::shared_ptr<CNode<D, R>>> ca;
    std::optional<std::shared_ptr<CNode<D, R>>> cfilter;
    std::optional<std::shared_ptr<CNode<D, 1>>> cbias;
    int stride, padding;


    template <typename OtherDerived, typename OtherDerived2>
    static auto myConvolution(const OtherDerived &a, const OtherDerived2 &filter, int padding, int stride = 1, int dialation = 1, int additionalBRPadding = 0) {

        auto patches = a.extract_image_patches(filter.dimension(1), filter.dimension(2), stride, stride, 1, 1, dialation, dialation, padding, padding + additionalBRPadding, padding, padding + additionalBRPadding, 0);

        int sheight = (dialation * (a.dimension(1) - 1) + 1 - filter.dimension(1) + 2 * padding + additionalBRPadding) / stride + 1;
        int swidth = (dialation * (a.dimension(2) - 1) + 1 - filter.dimension(2) + 2 * padding + additionalBRPadding) / stride + 1;
        Eigen::array<long, R - 1> reshape{filter.dimension(0) * filter.dimension(1) * filter.dimension(2),
                                          sheight * swidth,
                                          a.dimension(3)};

        auto im2col = patches.reshape(reshape);
        auto i2cFilter = filter.reshape(Eigen::array<long, 2>{reshape[0], filter.dimension(3)});
        auto conv = i2cFilter.contract(im2col, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(0, 0)});

        Eigen::array<long, R> reshape2{filter.dimension(3), sheight, swidth, a.dimension(3)};
        return conv.reshape(reshape2);
    }

    static void myDialationForFilter(Eigen::Tensor<D, R-1> &out, const Eigen::Tensor<D, R-1> &in, int stride) {
        out.setConstant(0);
        #pragma omp parallel for
        for (int c = 0, ci = 0; c < out.dimension(2); c += stride, ci++)
            for (int b = 0, bi = 0; b < out.dimension(1); b += stride, bi++)
                for (int a = 0, ai = 0; a < out.dimension(0); a += stride, ai++)
                    out(a, b, c) = in(ai, bi, ci);
    }
};

#undef R

#endif //LIBDL_CONV2D_H