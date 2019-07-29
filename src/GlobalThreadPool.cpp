
#include "GlobalThreadPool.h"

Eigen::ThreadPool GlobalThreadPool::pool(32);
Eigen::ThreadPoolDevice GlobalThreadPool::myDevice(&GlobalThreadPool::pool, 32);
