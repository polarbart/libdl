//
// Created by polarbabe on 18.05.19.
//

#ifndef LIBDL_MATMUL_H
#define LIBDL_MATMUL_H

#include <memory>
#include "CNode.h"
#include "Tensor.h"

template <typename D, int R>
class MatMul : public CNode {
    public:
    void backward() override;

    static std::shared_ptr<Tensor<D, R>> matmul(Tensor<D, R>& a, Tensor<D, R>& b);

    MatMul(std::optional<std::shared_ptr<CNode>> a, std::optional<std::shared_ptr<CNode>> b, Eigen::Tensor<D, R>, std::weak_ptr<Tensor<D, R>>);

private:
    std::optional<std::shared_ptr<CNode>> a, b;
    std::weak_ptr<Tensor<D, R>> tensor;
};


#endif //LIBDL_MATMUL_H
