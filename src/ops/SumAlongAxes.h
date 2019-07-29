
#ifndef LIBDL_SUMALONGAXES_H
#define LIBDL_SUMALONGAXES_H


#include "../Tensor.h"
#include "../Utils.h"

template <typename D, int RA, int RB>
class SumAlongAxes : public CNode<D, RA - RB> {
public:
    SumAlongAxes(
            const std::optional<std::shared_ptr<CNode<D, RA>>> &cx,
            const std::shared_ptr<Tensor<D, RA - RB>> &result,
            const std::array<int, RB> &axes,
            const std::array<long, RA> &oldDimensions)
            : CNode<D, RA - RB>(Utils::removeOption<std::shared_ptr<CNodeBase>>({cx}), result),
            cx(cx),
            axes(axes),
            oldDimensions(oldDimensions) {}

    /*
     * \brief computes the sum along the given axes
     *
     * \param x a tensor of any shape
     * \param axes the axes along which the sum should be computed
     *
     * \return a new tensor containing the sum
     * */
    static std::shared_ptr<Tensor<D, RA - RB>> sum(
            const std::shared_ptr<Tensor<D, RA>> &x,
            std::array<int, RB> axes) {

        for (auto a : axes)
            if (a < 0 || a >= RA)
                throw std::invalid_argument("axis index out of range");

        std::array<long, RA - RB> newShape {};
        for (int i = 0, j = 0; i < (RA - RB); j++)
            if (notIn(j, axes))
                newShape[i++] = x->data->dimension(j);

        auto result = std::make_shared<Tensor<D, RA - RB>>(x->data->sum(axes), newShape);
        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<SumAlongAxes<D, RA, RB>>(x->gradFn, result, axes, x->data->dimensions()));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            std::array<int, RA> reshape;
            std::array<int, RA> broadcast;
            for (int i = 0; i < RA; i++) {
                if (notIn(i, axes)) {
                    reshape[i] = oldDimensions[i];
                    broadcast[i] = 1;
                } else {
                    reshape[i] = 1;
                    broadcast[i] = oldDimensions[i];
                }
            }
            cx.value()->addGrad(CNode<D, RA - RB>::grad->reshape(reshape).broadcast(broadcast));
        }
        CNode<D, RA - RB>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, RA>>> cx;
    std::array<int, RB> axes;
    std::array<long, RA> oldDimensions;

    static bool notIn(int a, const std::array<int, RB> &axes) {
        for (auto i : axes)
            if (a == i)
                return false;
        return true;
    }
};

#endif //LIBDL_SUMALONGAXES_H
