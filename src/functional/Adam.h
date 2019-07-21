
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
            if (parameter->eTensor->dimension(i) != m->eTensor->dimension(i) || parameter->eTensor->dimension(i) != v->eTensor->dimension(i))
                throw std::invalid_argument("the shapes of parameter, m and v must match");

        // #efficient
        static Eigen::ThreadPool pool(8);
        static Eigen::ThreadPoolDevice myDevice(&pool, 8);

        m->eTensor->device(myDevice) = m->eTensor->constant(b1) * *m->eTensor + m->eTensor->constant(1 - b1) * *parameter->grad;
        v->eTensor->device(myDevice) = v->eTensor->constant(b2) * *v->eTensor + v->eTensor->constant(1 - b2) * parameter->grad->square();

        auto mh = *m->eTensor / m->eTensor->constant(1 - b1);
        auto vh = *v->eTensor / v->eTensor->constant(1 - b2);
        parameter->eTensor->device(myDevice) -= mh.constant(lr) * mh / (vh.sqrt() + vh.constant(eps));
    }

};


#endif //LIBDL_ADAM_H
