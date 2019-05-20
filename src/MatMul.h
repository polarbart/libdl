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
    void backward() override {};
    MatMul(std::shared_ptr<Tensor<D, R>> a, std::shared_ptr<Tensor<D, 2>> b, std::weak_ptr<Tensor<D, R>> operation) : a(a), b(b), operation(operation) {}

    static std::shared_ptr<Tensor<D, R>> matmul(std::shared_ptr<Tensor<D, R>> a,std::shared_ptr<Tensor<D, 2>> b) {
        std::array<long, R> shape {};
        std::copy_n(a->eTensor.dimensions().begin(), R, shape.begin());
        shape[R-1] = b->eTensor.dimension(1);
        long size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
        auto data = std::shared_ptr<D[]>(new D[size]);
        Eigen::TensorMap<Eigen::Tensor<D, R>> t(data.get(), shape);
        t = a->eTensor.contract(b->eTensor, Eigen::array<Eigen::IndexPair<int>, 1>{Eigen::IndexPair<int>(R-1, 0)});
        auto result = std::make_shared<Tensor<D, R>>(data, shape);
        if (a->needsGradient() || b->needsGradient())
            result->setGradFn(std::make_shared<MatMul<D, R>>(a, b, result));
        return result;
    }


private:
    std::shared_ptr<Tensor<D, R>> a;
    std::shared_ptr<Tensor<D, 2>> b;
    std::weak_ptr<Tensor<D, R>> operation;

};

#endif //LIBDL_MATMUL_H
