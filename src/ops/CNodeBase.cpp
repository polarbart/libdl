
#include "CNodeBase.h"

CNodeBase::CNodeBase(std::vector<std::shared_ptr<CNodeBase>> p) : parents(std::move(p)) {}

bool CNodeBase::noGrad = false;
