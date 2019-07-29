
#ifndef LIBDL_ADAM_H
#define LIBDL_ADAM_H

#include "../Tensor.h"

template<typename D, int R>
class Adam {
public:

    /*
     * \brief Computations for Adam
     *
     * \param parameter the parameter on which Adam should be performed, any shape
     * \param m running average of gradients, same shape as parameter
     * \param v running average of squared gradients, same shape as parameter
     * \param lr learning rate
     * \param b1 beta one
     * \param b2 beta two
     * \param eps epsilon
     *
     * */
    static void applyAdam(
            const std::shared_ptr<Tensor<D, R>> &parameter,
            const std::shared_ptr<Tensor<D, R>> &m,
            const std::shared_ptr<Tensor<D, R>> &v,
            D lr,
            D b1,
            D b2,
            D eps) {

        if (parameter->grad.use_count() == 0)
            return;

        if (b1 < 0 || b2 < 0 || eps < 0)
            throw std::invalid_argument("beta1, beta2 and epsilon must not be negative");

        for (int i = 0; i < R; i++)
            if (parameter->data->dimension(i) != m->data->dimension(i) || parameter->data->dimension(i) != v->data->dimension(i))
                throw std::invalid_argument("the shapes of parameter m and v must match");

        // #efficient
        m->data->device(GlobalThreadPool::myDevice) = m->data->constant(b1) * *m->data + m->data->constant(1 - b1) * *parameter->grad;
        v->data->device(GlobalThreadPool::myDevice) = v->data->constant(b2) * *v->data + v->data->constant(1 - b2) * parameter->grad->square();

        auto mh = *m->data / m->data->constant(1 - b1);
        auto vh = *v->data / v->data->constant(1 - b2);
        parameter->data->device(GlobalThreadPool::myDevice) -= mh.constant(lr) * mh / (vh.sqrt() + vh.constant(eps));
    }

};


#endif //LIBDL_ADAM_H
