
#ifndef LIBDL_SUMALONGAXES_H
#define LIBDL_SUMALONGAXES_H


#include "../Tensor.h"
#include "../Utils.h"

template <typename D,std::int64_t RA,std::int64_t RB>
class SumAlongAxes : public CNode<D, RA - RB> {
public:
    SumAlongAxes(
            const std::optional<std::shared_ptr<CNode<D, RA>>> &cx,
            const std::shared_ptr<Tensor<D, RA - RB>> &result,
            const std::array <std::int64_t, RB> &axes,
            const std::array<std::int64_t, RA> &oldDimensions)
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
            std::array <std::int64_t, RB> axes) {

        for (auto a : axes)
            if (a < 0 || a >= RA)
                throw std::invalid_argument("axis index out of range");

        std::array<std::int64_t, RA - RB> newShape {};
        for (std::int64_t i = 0, j = 0; i < (RA - RB); j++)
            if (notIn(j, axes))
                newShape[i++] = x->data->dimension(j);

        auto result = std::make_shared<Tensor<D, RA - RB>>(x->data->sum(axes), newShape);
        if (x->needsGradient() && !CNodeBase::noGrad)
            result->setGradFn(std::make_shared<SumAlongAxes<D, RA, RB>>(x->gradFn, result, axes, x->data->dimensions()));
        return result;
    }

    void computeGradients() override {
        if (cx.has_value()) {
            std::array <std::int64_t, RA> reshape;
            std::array <std::int64_t, RA> broadcast;
            for (std::int64_t i = 0; i < RA; i++) {
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
    std::array <std::int64_t, RB> axes;
    std::array<std::int64_t, RA> oldDimensions;

    static bool notIn(std::int64_t a, const std::array <std::int64_t, RB> &axes) {
        for (auto i : axes)
            if (a == i)
                return false;
        return true;
    }
};

#endif //LIBDL_SUMALONGAXES_H
