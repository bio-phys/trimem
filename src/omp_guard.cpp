/** \file omp_guard.cpp
 */
#include <utility>
#include <iostream>

#include "omp_guard.h"

namespace trimem {
 
// construct
OmpGuard::OmpGuard(omp_lock_t& lock) : lock_(&lock), owner_(false)
{
    test();
}

// destruct
OmpGuard::~OmpGuard()
{
    release();
}

// move (take ownership away from o in case)
OmpGuard::OmpGuard(OmpGuard&& o) :
    lock_(o.lock_),
    owner_(std::exchange(o.owner_, false)) {}

// unset the lock
void OmpGuard::release()
{
    if (owner_)
    {
        owner_ = false;
        omp_unset_lock(lock_);
    }
}

// test the lock
bool OmpGuard::test()
{
    if (!owner_)
    {
        owner_ = omp_test_lock(lock_);
    }
    return owner_;
}
 
}
