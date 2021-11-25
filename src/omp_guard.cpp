/** \file omp_guard.cpp
 */
#include "omp_guard.h"

namespace trimem {
 
// construct
OmpGuard::OmpGuard()
{
    omp_init_nest_lock(&lock_);
}

// destruct
OmpGuard::~OmpGuard()
{
    omp_destroy_nest_lock(&lock_);
}

// unset
void OmpGuard::release()
{
    // check if owned and release in case
    int state = omp_test_nest_lock(&lock_);
    if (state > 0)
    {
        for (int i=state; i>0; i--) omp_unset_nest_lock(&lock_);
    }
}

// test: try to lock but go on in any case
bool OmpGuard::test()
{
    int state = omp_test_nest_lock(&lock_);
    if (state == 1) return true;
    if (state > 1)
    {
        // no need to keep nested locks
        omp_unset_nest_lock(&lock_);
        return false;
    }
    return false;
}
 
}
