
#ifndef LIBDL_CONV2D_H
#define LIBDL_CONV2D_H

#include "CNode.h"
#include "../Utils.h"

#define R 4

template<typename D>
class Conv2D : public CNode<D, R> {
public:

    Conv2D(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &filter,
            const std::shared_ptr<Tensor<D, 1>> &bias,
            const std::shared_ptr<Tensor<D, R>> &result,
           std::int64_t padding,
           std::int64_t stride)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, filter->gradFn, bias->gradFn}), result),
            x(x->data),
            filter(filter->data),
            cx(x->gradFn),
            cfilter(filter->gradFn),
            cbias(bias->gradFn),
            padding(padding),
            stride(stride) {}

    Conv2D(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &filter,
            const std::shared_ptr<Tensor<D, R>> &result,
           std::int64_t padding,
           std::int64_t stride)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn, filter->gradFn}), result),
            x(x->data),
            filter(filter->data),
            cx(x->gradFn),
            cfilter(filter->gradFn),
            cbias(std::nullopt),
            padding(padding),
            stride(stride) {}

    /* \brief computes the convolution between the image like tensor x and filter. If bias is not null it is added (tied)
     *
     * \param x a 4d tensor of shape (c, h, w, batchsize)
     * \param filter a 4d tensor of shape (c, a, b, c')
     * \param bias a 1d tensor of shape (c',), may be null
     * \param padding the padding applied to the second and third dimension of tensor x
     * \param stride the stride used for convolution
     *
     * \return a new tensor with shape (c', (h - a + 2 * padding) / stride + 1, (w - b + 2 * padding) / stride + 1, batchsize)
     * */
    static std::shared_ptr<Tensor<D, R>> conv2d(
            const std::shared_ptr<Tensor<D, R>> &x,
            const std::shared_ptr<Tensor<D, R>> &filter,
            const std::shared_ptr<Tensor<D, 1>> &bias,
           std::int64_t padding,
            std::int64_t stride) {

        if (x->data->dimension(0) != filter->data->dimension(0) || (bias != nullptr && filter->data->dimension(3) != bias->data->dimension(0)))
            throw std::invalid_argument("the first dimensions of x, filter and bias must match");
        if (bias != nullptr && filter->data->dimension(3) != bias->data->dimension(0))
            throw std::invalid_argument("the last dimension of filter and the first dimension of bias must match");
        if (padding < 0)
            throw std::invalid_argument("padding must not be negative");
        if (stride <= 0)
            throw std::invalid_argument("stride must be positive");

        std::array<std::int64_t, R> newDims{
                filter->data->dimension(3),
                (x->data->dimension(1) - filter->data->dimension(1) + 2 * padding) / stride + 1,
                (x->data->dimension(2) - filter->data->dimension(2) + 2 * padding) / stride + 1,
                x->data->dimension(3),
        };

        if (newDims[1] <= 0 || newDims[2] <= 0)
            throw std::invalid_argument("the size of the resulting convolution is <= 0");

        auto r = myConvolution(*x->data, *filter->data, filter->data->dimensions(), padding, stride);

        std::shared_ptr<Tensor<D, R>> result;
        if (bias != nullptr) {
            Eigen::array<std::int64_t, R> reshape{bias->data->dimension(0), 1, 1, 1};
            Eigen::array<std::int64_t, R> broadcast{1, newDims[1], newDims[2], newDims[3]};
            result = std::make_shared<Tensor<D, R>>(r + bias->data->reshape(reshape).broadcast(broadcast), newDims);
        } else
            result = std::make_shared<Tensor<D, R>>(r, newDims);

        if ((x->needsGradient() || filter->needsGradient() || (bias != nullptr && bias->needsGradient())) && !CNodeBase::noGrad) {
            if (bias != nullptr)
                result->setGradFn(std::make_shared<Conv2D<D>>(x, filter, bias, result, padding, stride));
            else
                result->setGradFn(std::make_shared<Conv2D<D>>(x, filter, result, padding, stride));
        }
        return result;
    }

    void computeGradients() override {

        if (cx.has_value()) {

            std::int64_t temp = static_cast<std::int64_t>(cx.value()->shape[1] + filter->dimension(1) - stride * (CNode<D, R>::grad->dimension(1) - 1) - 2);
            std::int64_t backPadding = temp / 2;
            std::int64_t additionalBRPadding = temp % 2;

            auto backFilter = filter->reverse(Eigen::array<bool, R> {false, true, true, false}).shuffle(Eigen::array <std::int64_t, 4> {3, 1, 2, 0}).eval();

            std::array<std::int64_t, R> filterDims {filter->dimension(3), filter->dimension(1), filter->dimension(2), filter->dimension(0)};
            cx.value()->addGrad(myConvolution(*CNode<D, R>::grad, backFilter, filterDims, backPadding, 1, stride, additionalBRPadding));
        }
        if (cfilter.has_value()) {

            std::int64_t dheight = x->dimension(1) - filter->dimension(1) + 2 * padding + 1;
            std::int64_t dwidth = x->dimension(2) - filter->dimension(2) + 2 * padding + 1;

            Eigen::array<std::int64_t, R + 1> reshape{1, x->dimension(0), x->dimension(1), x->dimension(2), x->dimension(3)};
            Eigen::array<std::int64_t, R - 1> reshape2{dheight * dwidth, x->dimension(0) * filter->dimension(1) * filter->dimension(2), x->dimension(3)};
            Eigen::array<std::int64_t, R + 1> reshape3{filter->dimension(0), filter->dimension(1), filter->dimension(2), x->dimension(3), filter->dimension(3)};

            auto xvol = x->reshape(reshape).extract_volume_patches(1, dheight, dwidth, 1, 1, 1, 1, 1, 1, 0, 0, padding, padding, padding, padding, 0);

            #ifdef _WIN32  // otherwise compilation takes hours (for some weired reason)
                auto im2col = xvol.reshape(reshape2).eval();
            #else
                auto im2col = xvol.reshape(reshape2);
            #endif

            if (stride == 1) {
                auto i2cFilter = CNode<D, R>::grad->sum(Eigen::array <std::int64_t, 1>{3}).reshape(Eigen::array<std::int64_t, 2>{CNode<D, R>::grad->dimension(0), reshape2[0]}).eval();
                auto conv = im2col.contract(i2cFilter, Eigen::array<Eigen::IndexPair <std::int64_t>, 1> {Eigen::IndexPair <std::int64_t>(0, 1)});
                cfilter.value()->addGrad(conv.reshape(reshape3).mean(Eigen::array <std::int64_t, 1> {3}));
            } else {

                Eigen::Tensor<D, R - 1> dialated(CNode<D, R>::grad->dimension(0), dheight, dwidth);
                Eigen::Tensor<D, R - 1> summedGrad(CNode<D, R>::grad->dimension(0), CNode<D, R>::grad->dimension(1), CNode<D, R>::grad->dimension(2));
                summedGrad.device(GlobalThreadPool::myDevice) = CNode<D, R>::grad->sum(Eigen::array <std::int64_t, 1> {3});
                myDilationForFilter(dialated, summedGrad, stride);

                auto i2cFilter = dialated.reshape(Eigen::array<std::int64_t, 2>{CNode<D, R>::grad->dimension(0), reshape2[0]});
                auto conv = im2col.contract(i2cFilter, Eigen::array<Eigen::IndexPair <std::int64_t>, 1> {Eigen::IndexPair <std::int64_t>(0, 1)});

                cfilter.value()->addGrad(conv.reshape(reshape3).mean(Eigen::array <std::int64_t, 1> {3}));
            }
        }
        if (cbias.has_value()) {
            cbias.value()->addGrad(CNode<D, R>::grad->sum(Eigen::array <std::int64_t, 3> {1, 2, 3}));
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    std::shared_ptr<Eigen::Tensor<D, R>> x;
    std::shared_ptr<Eigen::Tensor<D, R>> filter;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    std::optional<std::shared_ptr<CNode<D, R>>> cfilter;
    std::optional<std::shared_ptr<CNode<D, 1>>> cbias;
    std::int64_t stride, padding;

    /*
     * \brief computes the convolution between x and filter with the im2col method
     *        this is by two orders of magnitude faster than the eigen implementation
     *
     * \param x
     * \param filter
     * \param padding
     * \param stride
     * \param dilation
     * \param additionalBRPadding additional padding put to the bottom right of x
     *
     * \brief returns the convolution between x and filter
     * */
    template<typename OtherDerived, typename OtherDerived2>
    static auto myConvolution(
            const OtherDerived &x,
            const OtherDerived2 &filter,
            const std::array<std::int64_t, R> filterDims,
            std::int64_t padding,
            std::int64_t stride = 1,
            std::int64_t dilation = 1,
            std::int64_t additionalBRPadding = 0) {

        auto patches = x.extract_image_patches(filterDims[1], filterDims[2], stride, stride, 1, 1,
                                               dilation, dilation, padding, padding + additionalBRPadding, padding,
                                               padding + additionalBRPadding, 0);

        std::int64_t sheight = (dilation * (x.dimension(1) - 1) + 1 - filterDims[1] + 2 * padding + additionalBRPadding) / stride + 1;
        std::int64_t swidth = (dilation * (x.dimension(2) - 1) + 1 - filterDims[2] + 2 * padding + additionalBRPadding) / stride + 1;
        Eigen::array<std::int64_t, R - 1> reshape{filterDims[0] * filterDims[1] * filterDims[2], sheight * swidth, x.dimension(3)};

        #ifdef _WIN32  // otherwise compilation takes hours (for some weired reason)
            auto im2col = patches.reshape(reshape).eval();
        #else
            auto im2col = patches.reshape(reshape);
        #endif
        auto i2cFilter = filter.reshape(Eigen::array<std::int64_t, 2>{reshape[0], filterDims[3]});
        auto conv = i2cFilter.contract(im2col, Eigen::array<Eigen::IndexPair <std::int64_t>, 1>{Eigen::IndexPair <std::int64_t>(0, 0)});

        Eigen::array<std::int64_t, R> reshape2{filterDims[3], sheight, swidth, x.dimension(3)};
        return conv.reshape(reshape2);
    }

    /*a
     * \brief dilates in and stores the result in out
     *
     * \param out
     * \param in
     * \param dilation
     * */
    static void myDilationForFilter(
            Eigen::Tensor<D, R - 1> &out,
            const Eigen::Tensor<D, R - 1> &in,
            std::int64_t dilation) {

        out.setZero();

        for (std::int64_t c = 0, ci = 0; c < out.dimension(2); c += dilation, ci++)
            for (std::int64_t b = 0, bi = 0; b < out.dimension(1); b += dilation, bi++)
                #pragma omp parallel for
                for (std::int64_t a = 0; a < out.dimension(0); a++)
                    out(a, b, c) = in(a, bi, ci);
    }
};

#undef R

#endif //LIBDL_CONV2D_H