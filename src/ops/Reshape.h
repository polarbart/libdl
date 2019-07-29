
#ifndef LIBDL_RESHAPE_H
#define LIBDL_RESHAPE_H


#include "CNode.h"
#include "../Utils.h"

template <typename D, int RA, int RB>
class Reshape : public CNode<D, RB> {

public:
    Reshape(
            const std::shared_ptr<Tensor<D, RA>> &x,
            const std::shared_ptr<Tensor<D, RB>> &result)
            : CNode<D, RB>(Utils::removeOption<std::shared_ptr<CNodeBase>>({x->gradFn}), result),
            oldShape(x->data->dimensions()),
            cx(x->gradFn) {}

    /*
     * \brief reshapes the given tensor to the given shape
     *
     * \param x a tensor of any shape that should be reshaped
     * \param newShape the shape to which x should be reshaped
     *                 one element can be -1, its value is then infered from the size of x and the remaining dimensions
     *
     * \return a new tensor with the new shape
     * */
    static std::shared_ptr<Tensor<D, RB>> reshape(
            const std::shared_ptr<Tensor<D, RA>> &x,
            std::array<long, RB> newShape) {

        for (int i = 0; i < RB; i++)
            if (newShape[i] == -1) {
                newShape[i] = x->data->size() / std::accumulate(std::begin(newShape), std::end(newShape), -1, std::multiplies<>());
                break;
            }

        int newSize = std::accumulate(std::begin(newShape), std::end(newShape), 1, std::multiplies<>());
        if (newSize != x->data->size())
            throw std::invalid_argument("x can't be reshaped to the given shape");

        auto result = std::make_shared<Tensor<D, RB>>(x->data->reshape(newShape), newShape);
        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<Reshape<D, RA, RB>>(x, result));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value())
            cx.value()->addGrad(CNode<D, RB>::grad->reshape(oldShape));
        CNode<D, RB>::finishComputeGradient();
    }

private:
    std::array<long, RA> oldShape;
    std::optional<std::shared_ptr<CNode<D, RA>>> cx;
};


#endif //LIBDL_RESHAPE_H
