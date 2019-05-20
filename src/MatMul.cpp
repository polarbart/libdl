#include "MatMul.h"
#include <numeric>
/*
template<typename D, int R>
std::shared_ptr<Tensor<D, R>> MatMul<D, R>::matmul(std::shared_ptr<Tensor<D, R>> a, std::shared_ptr<Tensor<D, 2>> b) {
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

template<typename D, int R>
MatMul<D, R>::MatMul(std::shared_ptr<Tensor<D, R>> a, std::shared_ptr<Tensor<D, 2>> b, std::weak_ptr<Tensor<D, R>> operation)
        : a(a), b(b), operation(operation) {}
*/
template <typename D, int R>
void init_MatMul(py::module &m) {
    m.def("matmul", &MatMul<D, R>::matmul);
}