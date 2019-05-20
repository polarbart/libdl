//
// Created by polarbabes on 16.05.19.
//

#ifndef LIBDL_ADD_H
#define LIBDL_ADD_H

#include "Tensor.h"

template <typename D, int R>
class Add : public CNode {

public:
    void backward() override {};
    Add(std::optional<std::shared_ptr<CNode>> a, std::optional<std::shared_ptr<CNode>> b, std::weak_ptr<Tensor<D, R>> tensor) : a(std::move(a)), b(std::move(b)), tensor(tensor) {};

    static std::shared_ptr<Tensor<D, R>> add(std::shared_ptr<Tensor<D, R>> a, std::shared_ptr<Tensor<D, R>> b) {
        auto data = std::shared_ptr<D[]>(new D[a->eTensor.size()]);
        Eigen::TensorMap<Eigen::Tensor<D, R>> t(data.get(), a->eTensor.dimensions());
        t = a->eTensor + b->eTensor;
        std::array<long, R> shape {};
        std::copy(t.dimensions().begin(), t.dimensions().end(), shape.begin());
        auto result = std::make_shared<Tensor<D, R>>(data, shape);
        if (a->needsGradient() || b->needsGradient())
            result->setGradFn(std::make_shared<Add<D, R>>(a->getGradFn(), b->getGradFn(), result));
        return result;
    }


private:
    std::optional<std::shared_ptr<CNode>> a, b;
    std::weak_ptr<Tensor<D, R>> tensor;

};


#endif //LIBDL_ADD_H
