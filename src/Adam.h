//
// Created by superbabes on 21.06.19.
//

#ifndef LIBDL_ADAM_H
#define LIBDL_ADAM_H

#include "Tensor.h"

template <typename D, int R>
class Adam {
public:
    static void applyAdam(
            const std::shared_ptr<Tensor<D, R>> &param,
            const std::shared_ptr<Tensor<D, R>> &m,
            const std::shared_ptr<Tensor<D, R>> &v,
            D lr,
            D b1,
            D b2,
            D eps) {
        *m->eTensor = m->eTensor->constant(b1) * *m->eTensor + m->eTensor->constant(1 - b1) * *param->grad;
        *v->eTensor = v->eTensor->constant(b2) * *v->eTensor + v->eTensor->constant(1 - b2) * param->grad->square();
        auto mh = *m->eTensor / m->eTensor->constant(1 - b1);
        auto vh = *v->eTensor / v->eTensor->constant(1 - b2);
        *param->eTensor -= mh.constant(lr) * mh / (vh.sqrt() + vh.constant(eps));
    }

};


#endif //LIBDL_ADAM_H
