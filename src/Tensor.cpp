#include <utility>
#include <iostream>

#include "Tensor.h"
//#include "Add.h"


template<typename D, int R>
Tensor<D, R>::Tensor(pybind11::array_t<D, py::array::f_style> data, bool requires_grad)
        : data(data),
          eTensor(intiTensorMap(data.request(true))),
          grad_fn(std::nullopt),
          requires_grad(requires_grad) {}

template<typename D, int R>
Tensor<D, R>::Tensor(Eigen::TensorMap<Eigen::Tensor<D, R>> &eTensor)
        : eTensor(eTensor),
          data(intiNumpyBuffer(eTensor)),
          grad_fn(std::nullopt),
          requires_grad(false) {}

template<typename D, int R>
Eigen::TensorMap<Eigen::Tensor<D, R>> Tensor<D, R>::intiTensorMap(const py::buffer_info bufferInfo) {
    std::vector<ssize_t> shape = bufferInfo.shape;
    Eigen::array<ssize_t, R> m;
    for (size_t i = 0; i < R; i++)
        m[i] = shape[i];
    return Eigen::TensorMap<Eigen::Tensor<D, R>>(static_cast<D *>(bufferInfo.ptr), m);
}

template<typename D, int R>
py::array_t<D, py::array::f_style> Tensor<D, R>::intiNumpyBuffer(Eigen::TensorMap<Eigen::Tensor<D, R>> &t) {
    auto shape = t.dimensions();
    size_t a[R], s[R];
    a[R - 1] = sizeof(D);
    s[R - 1] = shape[R - 1];
    for (int i = R - 2; i >= 0; i--) {
        a[i] = shape[i] * a[i + 1];
        s[i] = shape[i];
    }
    return py::array_t<D, py::array::f_style>(s, a, t.data());
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
std::shared_ptr<Tensor<D, R>> Tensor<D, R>::make_tensor(Eigen::TensorMap<Eigen::Tensor<D, R>> &r) {
    return std::shared_ptr<Tensor<D, R>>(new Tensor(r));
}

//#include "Add.h"
void init_Add(py::module &m);
PYBIND11_MODULE(libdl, m) {
    //py::class_<Eigen::Tensor<float, 2>>(m, "_EigenTensor").def(py::init());
    py::class_<Tensor<float, 2>>(m, "Tensor")
            .def(py::init<py::array_t<float, py::array::f_style>, bool>())
            // .def("_from_eigen", &Eigen::Tensor<float, 2>&>())
            .def("_make_tensor", &Tensor<float, 2>::make_tensor)
            .def_readonly("data", &Tensor<float, 2>::data)
            .def_readwrite("requires_grad", &Tensor<float, 2>::requires_grad)
            .def("setGradFn", &Tensor<float, 2>::setGradFn)
            .def("getGradFn", &Tensor<float, 2>::getGradFn)
            .def("needsGradient", &Tensor<float, 2>::needsGradient);
    //m.def("add", &Add<float, 2>::add, py::return_value_policy::take_ownership);
    //std::cout << "hgöi" << std::endl;
    init_Add(m);
}