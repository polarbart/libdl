#include <utility>

//
// Created by superbabes on 16.06.19.
//

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
            const std::shared_ptr<Tensor<D, R>> &result,
            int kernelSizeAndStride)
            : CNode<D, R>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result),
            argmax(std::move(argmax)),
            cx(x->gradFn),
            kernelSizeAndStride(kernelSizeAndStride) {}
    /*
     * x (c, h, w, n)
     * */
    static std::shared_ptr<Tensor<D, R>> maxpool2d(const std::shared_ptr<Tensor<D, R>> &x, int kernelSizeAndStride) {
        std::array<long, R> newShape {x->eTensor->dimension(0), x->eTensor->dimension(1) / kernelSizeAndStride, x->eTensor->dimension(2) / kernelSizeAndStride, x->eTensor->dimension(3)};
        Eigen::Tensor<int, R + 1> argmax(2, newShape[0], newShape[1], newShape[2], newShape[3]);
        auto result = std::make_shared<Tensor<D, R>>(newShape);

        #pragma omp parallel for
        for (int a = 0; a < x->eTensor->dimension(3); a++) { // batchsize
            for (int b = 0; b < x->eTensor->dimension(2) / kernelSizeAndStride * kernelSizeAndStride; b++) { // w
                int w = b / kernelSizeAndStride;
                int wr = b % kernelSizeAndStride;
                for (int c = 0; c < x->eTensor->dimension(1) / kernelSizeAndStride * kernelSizeAndStride; c++) { // h
                    int h = c / kernelSizeAndStride;
                    int hr = c % kernelSizeAndStride;

                    if (wr == 0 && hr == 0)
                        for (int d = 0; d < x->eTensor->dimension(0); d++) { // c
                            argmax(0, d, h, w, a) = 0;
                            argmax(1, d, h, w, a) = 0;
                            (*result->eTensor)(d, h, w, a) = (*x->eTensor)(d, c, b, a);
                        }
                    else
                        for (int d = 0; d < x->eTensor->dimension(0); d++) { // c
                            if ((*x->eTensor)(d, c, b, a) > (*result->eTensor)(d, h, w, a)) {
                                argmax(0, d, h, w, a) = hr;
                                argmax(1, d, h, w, a) = wr;
                                (*result->eTensor)(d, h, w, a) = (*x->eTensor)(d, c, b, a);
                            }
                        }
                }
            }
        }
        if (x->needsGradient())
            result->setGradFn(std::make_shared<MaxPool2D<D>>(x, std::move(argmax), result, kernelSizeAndStride));
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
