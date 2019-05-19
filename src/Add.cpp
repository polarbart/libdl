#include <utility>

#include <utility>

#include <utility>

//
// Created by polarbabes on 16.05.19.
//

#include "Add.h"

template<typename D, int R>
void Add<D, R>::backward() {

}

template<typename D, int R>
std::shared_ptr<Tensor<D, R>> Add<D, R>::add(Tensor<D, R> &a, Tensor<D, R> &b) {
    size_t s = 1;
    for (int i = 0; i < R; ++i) {
        s *= a.eTensor.dimension(i);
    }
    auto data = new D[s];
    Eigen::TensorMap<Eigen::Tensor<D, R>> t(data, a.eTensor.dimensions());
    t = a.eTensor + b.eTensor;
    auto result = Tensor<D, R>::make_tensor(t);
    // auto result = std::make_shared<Tensor<D, R>>(a.data);
    // auto result = std::shared_ptr(new Tensor<D, R>(a.eTensor + b.eTensor));
    if (a.needsGradient() || b.needsGradient())
        result->setGradFn(std::make_shared<Add<D, R>>(a.getGradFn(), b.getGradFn(), result));
    return result;
}

template<typename D, int R>
Add<D, R>::Add(std::optional<std::shared_ptr<CNode>> a, std::optional<std::shared_ptr<CNode>> b, std::weak_ptr<Tensor<D, R>> tensor) : a(std::move(a)), b(std::move(b)), tensor(tensor) {}

void init_Add(py::module &m) {
    m.def("add", &Add<float, 2>::add, py::return_value_policy::take_ownership);
}
