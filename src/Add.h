//
// Created by polarbabes on 16.05.19.
//

#ifndef LIBDL_ADD_H
#define LIBDL_ADD_H

#include "Tensor.h"

template <typename D, int R>
class Add : public CNode {

public:
    void backward() override;

    static std::shared_ptr<Tensor<D, R>> add(Tensor<D, R>& a, Tensor<D, R>& b);

    Add(std::optional<std::shared_ptr<CNode>> a, std::optional<std::shared_ptr<CNode>> b, std::weak_ptr<Tensor<D, R>>);

private:
    std::optional<std::shared_ptr<CNode>> a, b;
    std::weak_ptr<Tensor<D, R>> tensor;

};


#endif //LIBDL_ADD_H
