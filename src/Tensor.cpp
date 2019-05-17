#include <utility>

#include "Tensor.h"

template<typename D, int R>
Tensor<D, R>::Tensor(pybind11::array_t<D, py::array::f_style> data, bool requires_grad)
        : data(data),
          bufferInfo(data.request()),
          eTensor(intiTensorMap()),
          grad_fn(std::nullopt),
          requires_grad(requires_grad) {}

template<typename D, int R>
Tensor<D, R>::Tensor(const Eigen::Tensor<D, R> &eTensor)
        : eTensor(eTensor),
          data(intiNumpyBuffer()),
          bufferInfo(data.request()),
          grad_fn(std::nullopt),
          requires_grad(false) {}

template<typename D, int R>
const Eigen::TensorMap<Eigen::Tensor<D, R>> Tensor<D, R>::intiTensorMap() {
    std::vector<ssize_t> shape = bufferInfo.shape;
    this->requires_grad = true;
    Eigen::array<ssize_t, R> m;
    for (size_t i = 0; i < R; i++)
        m[i] = shape[i];
    return Eigen::TensorMap<Eigen::Tensor<D, R>>(static_cast<D *>(bufferInfo.ptr), m);
}

template<typename D, int R>
const py::array_t<D, py::array::f_style> Tensor<D, R>::intiNumpyBuffer() {
    auto shape = eTensor.dimensions().data();
    size_t a[R];
    a[R - 1] = sizeof(D);
    for (int i = R - 2; i >= 0; i--)
        a[i] = shape[i] * a[i + 1];
    return py::array_t<D, py::array::f_style>(eTensor.dimensions().data(), a, eTensor.data());
}

template<typename D, int R>
void Tensor<D, R>::setGradFn(std::shared_ptr<CNode> g) {
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

void init_Add(py::module &m);
PYBIND11_MODULE(libdl, m) {
    py::class_<Tensor<float, 2>>(m, "Tensor")
            .def(py::init<py::array_t<float, py::array::f_style>, bool>())
            .def_readonly("data", &Tensor<float, 2>::data)
            .def_readwrite("requires_grad", &Tensor<float, 2>::requires_grad);
    init_Add(m);
}
