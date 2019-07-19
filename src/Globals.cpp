//
// Created by superbabes on 19.07.19.
//

#include "Globals.h"

bool Globals::forceNoGrad;
Eigen::ThreadPool Globals::pool(8);
Eigen::ThreadPoolDevice Globals::myDevice(&Globals::pool, 8);
