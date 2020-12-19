
#ifndef LIBDL_ADD_H
#define LIBDL_ADD_H

#include "../Tensor.h"
#include "CNode.h"
#include "../Utils.h"

template<typename D,std::int64_t RA,std::int64_t RB>
class Add : public CNode<D, std::max<std::int64_t>(RA, RB)> {

public:
    Add(
            const std::optional<std::shared_ptr<CNode<D, RA>>> &ca,
            const std::optional<std::shared_ptr<CNode<D, RB>>> &cb,
            const std::shared_ptr<Tensor<D, std::max<std::int64_t>(RA, RB)>> &result)
            : CNode<D, std::max<std::int64_t>(RA, RB)>(Utils::removeOption<std::shared_ptr<CNodeBase>>({ca, cb}), result),
            ca(ca),
            cb(cb) {};


    /*
     * \brief Elementwise addition.
     *        One tensor x can have a less dimensions than the other tensor y, as long as the first dimensions match.
     *        x will be broadcasted to match y.
     *        E.g. shapes (10, 5) and (10, 5, 8) are ok, but shapes (10, 5) and (8, 10, 5) are not
     *
     * \param a tensor of any shape
     * \param b
     *
     * \return a new tensor of the shape with more dimensions
     * */
    static std::shared_ptr<Tensor<D, std::max<std::int64_t>(RA, RB)>> add(
            const std::shared_ptr<Tensor<D, RA>> &a,
            const std::shared_ptr<Tensor<D, RB>> &b) {

        if constexpr (RB > RA)
            return Add<D, RB, RA>::add(b, a);
        else {
            for (std::int64_t i = 0; i < RB; i++)
                if (a->data->dimension(i) != b->data->dimension(i))
                    throw std::invalid_argument("shapes mismatch");

            std::shared_ptr<Tensor<D, RA>> result;
            if constexpr (RB < RA) {
                std::array <std::int64_t, RA> reshape{};
                std::array <std::int64_t, RA> broadcast{};
                for (std::int64_t i = 0; i < RB; ++i) {
                    reshape[i] = b->data->dimension(i);
                    broadcast[i] = 1;
                }
                for (std::int64_t i = RB; i < RA; ++i) {
                    reshape[i] = 1;
                    broadcast[i] = a->data->dimension(i);
                }
                result = std::make_shared<Tensor<D, RA>>(
                        *a->data + b->data->reshape(reshape).broadcast(broadcast), a->data->dimensions());
            } else
                result = std::make_shared<Tensor<D, RA>>(*a->data + *b->data, a->data->dimensions());

            if ((a->needsGradient() || b->needsGradient()) && !CNodeBase::noGrad)
                result->setGradFn(std::make_shared<Add<D, RA, RB>>(a->gradFn, b->gradFn, result));
            return result;
        }
    }

    void computeGradients() override {
        if (ca.has_value())
            ca.value()->addGrad(*CNode<D, std::max<std::int64_t>(RA, RB)>::grad);
        if (cb.has_value()) {
            if constexpr (RB < RA) {
                std::array <std::int64_t, RA - RB> sumDims {};
                for (std::int64_t i = 0; i < RA - RB; ++i)
                    sumDims[i] = RB + i;
                cb.value()->addGrad(CNode<D, std::max<std::int64_t>(RA, RB)>::grad->sum(sumDims));
            } else
                cb.value()->addGrad(*CNode<D, std::max<std::int64_t>(RA, RB)>::grad);
        }
        CNode<D, std::max<std::int64_t>(RA, RB)>::finishComputeGradient();
    }

private:
    std::optional<std::shared_ptr<CNode<D, RA>>> ca;
    std::optional<std::shared_ptr<CNode<D, RB>>> cb;
};


#endif //LIBDL_ADD_H
