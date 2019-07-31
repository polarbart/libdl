
#ifndef LIBDL_CNODEBASE_H
#define LIBDL_CNODEBASE_H


#include <vector>
#include <memory>
#include <stdexcept>

/*
 * Just the baseclass of CNode since the template parameter D and R are not always known
 * */
class CNodeBase {
public:
    std::vector<std::shared_ptr<CNodeBase>> parents;
    int childrenThatNeedToComputeGradients = 0;
    bool visited = false;

    // compute the gradient of all parents and set them via CNode<D, R>::addGrad()
    virtual void computeGradients() = 0;

    // indicator if gradients should be disabled globally
    static bool noGrad;
protected:
    explicit CNodeBase(std::vector<std::shared_ptr<CNodeBase>>);
};


#endif //LIBDL_CNODEBASE_H
