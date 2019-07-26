
#ifndef LIBDL_MAXPOOL2D_H
#define LIBDL_MAXPOOL2D_H


#include "CNode.h"
#include "../Utils.h"
#define R 4

template <typename D>
class MaxPool2D : public CNode<D, R> {

public:

    MaxPool2D(
            const std::shared_ptr<Tensor<D, R>> &x,
            Eigen::Tensor<int, R + 1> argmax,
            int kernelSizeAndStride,
            const std::shared_ptr<Tensor<D, R>> &result)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result),
            argmax(std::move(argmax)),
            cx(x->gradFn),
            kernelSizeAndStride(kernelSizeAndStride) {}

    /*
     * \brief performs maxpooling on the image like tensor
     *
     * \param x a 4d tensor of shape (c, h, w, batchsize)
     * \param kernelSizeAndStride the size of the kernel which is also the stride
     *
     * \return a new tensor of shape (c, h/kernelSizeAndStride, w/kernelSizeAndStride, batchsize)
     * */
    static std::shared_ptr<Tensor<D, R>> maxpool2d(
            const std::shared_ptr<Tensor<D, R>> &x,
            int kernelSizeAndStride) {

        if (kernelSizeAndStride <= 0)
            throw std::invalid_argument("kernelSizeAndStride must be positive");

        std::array<long, R> newShape {
            x->data->dimension(0),
            x->data->dimension(1) / kernelSizeAndStride,
            x->data->dimension(2) / kernelSizeAndStride,
            x->data->dimension(3)
        };
        Eigen::Tensor<int, R + 1> argmax(2, newShape[0], newShape[1], newShape[2], newShape[3]);

        auto result = std::make_shared<Tensor<D, R>>(newShape);

        #pragma omp parallel for
        for (int a = 0; a < x->data->dimension(3); a++) { // batchsize
            for (int b = 0; b < x->data->dimension(2) - (x->data->dimension(2) % kernelSizeAndStride); b++) { // w
                int w = b / kernelSizeAndStride;
                int wr = b % kernelSizeAndStride;
                for (int c = 0; c < x->data->dimension(1) - (x->data->dimension(1) % kernelSizeAndStride); c++) { // h
                    int h = c / kernelSizeAndStride;
                    int hr = c % kernelSizeAndStride;

                    if (wr == 0 && hr == 0)
                        for (int d = 0; d < x->data->dimension(0); d++) { // c
                            argmax(0, d, h, w, a) = 0;
                            argmax(1, d, h, w, a) = 0;
                            (*result->data)(d, h, w, a) = (*x->data)(d, c, b, a);
                        }
                    else
                        for (int d = 0; d < x->data->dimension(0); d++) { // c
                            if ((*x->data)(d, c, b, a) > (*result->data)(d, h, w, a)) {
                                argmax(0, d, h, w, a) = hr;
                                argmax(1, d, h, w, a) = wr;
                                (*result->data)(d, h, w, a) = (*x->data)(d, c, b, a);
                            }
                        }
                }
            }
        }
        if (x->needsGradient())
            result->setGradFn(std::make_shared<MaxPool2D<D>>(x, std::move(argmax), kernelSizeAndStride, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            cx.value()->emptyGrad();
            cx.value()->grad->setZero();

            #pragma omp parallel for
            for (int a = 0; a < argmax.dimension(4); a++) // batchsize
                for (int b = 0; b < argmax.dimension(3); b++) // w
                    for (int c = 0; c < argmax.dimension(2); c++) // h
                        for (int d = 0; d < argmax.dimension(1); d++) { // c
                            int h = c * kernelSizeAndStride + argmax(0, d, c, b, a);
                            int w = b * kernelSizeAndStride + argmax(1, d, c, b, a);
                            (*cx.value()->grad)(d, h, w, a) = (*CNode<D, R>::grad)(d, c, b, a);
                        }
        }
        CNode<D, R>::finishComputeGradient();
    }

private:
    Eigen::Tensor<int, R + 1> argmax;
    std::optional<std::shared_ptr<CNode<D, R>>> cx;
    int kernelSizeAndStride;
};

#undef R

#endif //LIBDL_MAXPOOL2D_H
