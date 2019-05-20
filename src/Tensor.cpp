#include <utility>
#include <iostream>

#include "Tensor.h"
#include "MatMul.h"
#include "Add.h"

template <typename D, int R>
Tensor<D, R>::Tensor(std::shared_ptr<D[]> d, std::array<long, R>& shape, bool requiresGrad)
        : data(initNpArray(d.get(), shape)),
          eTensor(Eigen::TensorMap<Eigen::Tensor<D, R>>(d.get(), shape)),
          grad_fn(std::nullopt),
          requires_grad(requiresGrad),
          iData(d) {}

template<typename D, int R>
py::array_t<D, py::array::f_style> Tensor<D, R>::initNpArray(D* d, std::array<long, R>& shape) {
    size_t strides[R];
    strides[0] = sizeof(D);
    for (int i = 1; i < R; ++i)
        strides[i] = shape[i] * strides[i - 1]; // TODO right?
    return py::array_t<D, py::array::f_style>(shape, strides, d);
}

template<typename D, int R>
std::shared_ptr<Tensor<D, R>> Tensor<D, R>::fromNumpy(py::array_t<D, py::array::f_style> array, bool requiresGrad) {
    auto info = array.request(false);
    auto d = std::shared_ptr<D[]>(new D[info.size]);
    std::copy_n(static_cast<D*>(info.ptr), info.size, d.get());
    std::array<long, R> shape {};
    for (int i = 0; i < R; ++i)
        shape[i] = info.shape[i];
    return std::shared_ptr<Tensor<D, R>>(new Tensor<D, R>(d, shape, requiresGrad));
}

template<typename D, int R>
void Tensor<D, R>::setGradFn(const std::shared_ptr<CNode>& g) {
    grad_fn = std::optional<std::shared_ptr<CNode>>(g);
}

template<typename D, int R>
std::optional<std::shared_ptr<CNode>> Tensor<D, R>::getGradFn() {
    return grad_fn;
}

template<typename D, int R>
bool Tensor<D, R>::needsGradient() {
    return requires_grad || grad_fn.has_value();
}

template<typename D, int R>
void init_Module(py::module &m) {
    std::string name = "Tensor" + std::to_string(R);
    py::class_<Tensor<D, R>, std::shared_ptr<Tensor<D, R>>>(m, name.c_str())
            .def(py::init(&Tensor<D, R>::fromNumpy))
            .def_readonly("data", &Tensor<D, R>::data)
            .def_readwrite("requires_grad", &Tensor<D, R>::requires_grad)
            .def("setGradFn", &Tensor<D, R>::setGradFn)
            .def("getGradFn", &Tensor<D, R>::getGradFn)
            .def("needsGradient", &Tensor<D, R>::needsGradient);
    m.def("add", &Add<D, R>::add);
    m.def("matmul", &MatMul<D, R>::matmul);
}
PYBIND11_MODULE(libdl, m) {
    init_Module<float, 1>(m);
    init_Module<float, 2>(m);
    init_Module<float, 3>(m);
    init_Module<float, 4>(m);
    init_Module<float, 5>(m);
    /*
    init_Module<double, 1>(m);
    init_Module<double, 2>(m);
    init_Module<double, 3>(m);
    init_Module<double, 4>(m);
    init_Module<double, 5>(m);
     */
}