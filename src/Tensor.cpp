#include <utility>
#include <iostream>

#include "Tensor.h"
//#include "Add.h"

template <typename D, int R>
Tensor<D, R>::Tensor(std::shared_ptr<D[]> d, std::array<long, R>& shape, bool requiresGrad)
        : data(initNpArray(d, shape)),
          eTensor(Eigen::TensorMap<Eigen::Tensor<D, R>>(d.get(), shape)),
          grad_fn(std::nullopt),
          requires_grad(requiresGrad),
          iData(d) {}

template<typename D, int R>
Eigen::TensorMap<Eigen::Tensor<D, R>> Tensor<D, R>::intiTensorMap(const py::buffer_info bufferInfo) {
    std::vector<ssize_t> shape = bufferInfo.shape;
    Eigen::array<ssize_t, R> m;
    for (size_t i = 0; i < R; i++)
        m[i] = shape[i];
    return Eigen::TensorMap<Eigen::Tensor<D, R>>(static_cast<D *>(bufferInfo.ptr), m);
}

template<typename D, int R>
py::array_t<D, py::array::f_style> Tensor<D, R>::initNpArray(std::shared_ptr<D[]> d, std::array<long, R>& shape) {
    size_t strides[R];
    strides[R - 1] = sizeof(D);
    for (int i = R - 2; i >= 0; --i)
        strides[i] = shape[i] * strides[i + 1];
    return py::array_t<D, py::array::f_style>(shape, strides, d.get());
}

template<typename D, int R>
Tensor<D, R>* Tensor<D, R>::fromNumpy(py::array_t<D, py::array::f_style> array, bool requiresGrad) {
    auto info = array.request(false);
    auto d = std::shared_ptr<D[]>(new D[info.size]);
    std::copy_n(static_cast<D*>(info.ptr), info.size, d.get());
    std::array<long, R> shape {};
    for (int i = 0; i < R; ++i)
        shape[i] = info.shape[i];
    return new Tensor<D, R>(d, shape, requiresGrad);
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

//#include "Add.h"
void init_Add(py::module &m);
PYBIND11_MODULE(libdl, m) {
    //py::class_<Eigen::Tensor<float, 2>>(m, "_EigenTensor").def(py::init());
    py::class_<Tensor<float, 2>>(m, "Tensor")
            .def(py::init(&Tensor<float, 2>::fromNumpy))
            // .def("_from_eigen", &Eigen::Tensor<float, 2>&>())
            .def_readonly("data", &Tensor<float, 2>::data)
            .def_readwrite("requires_grad", &Tensor<float, 2>::requires_grad)
            .def("setGradFn", &Tensor<float, 2>::setGradFn)
            .def("getGradFn", &Tensor<float, 2>::getGradFn)
            .def("needsGradient", &Tensor<float, 2>::needsGradient);
    //m.def("add", &Add<float, 2>::add, py::return_value_policy::take_ownership);
    //std::cout << "hgöi" << std::endl;
    init_Add(m);
}