//
// Created by polarbabe on 12.05.19.
//

#ifndef LIBDL_CNODE_H
#define LIBDL_CNODE_H

#include <memory>
#include <vector>
#include <map>
#include <unsupported/Eigen/CXX11/Tensor>


class CNode {
public:
    void backward() {

    }

protected:
    virtual void computeGradients() = 0;
    int parentsThatNeedToComputeGradients = 0;

};


#endif //LIBDL_CNODE_H
