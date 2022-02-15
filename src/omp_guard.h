/** \file omp_guard.h
 * \brief Utility class to handle omp locks in a scoped way.
 *
 * Inspired by:
 * http://www.thinkingparallel.com/2006/08/21/...
 * ...scoped-locking-vs-critical-in-openmp-a-personal-shootout/
 */
#ifndef OMP_GUARD_H
#define OMP_GUARD_H
 
#include <omp.h>

namespace trimem {
 
class OmpGuard {
public:
    // construct
    OmpGuard(omp_lock_t& lock);
    ~OmpGuard();

    // don't allow copy
    OmpGuard(const OmpGuard&) = delete;
    OmpGuard operator=(const OmpGuard&) = delete;

    // move is supported
    OmpGuard(OmpGuard&& o);

    // use the guard
    bool test();
    void release();
 
private:
    omp_lock_t* lock_;
    bool owner_;
};
} 
#endif // OMP_GUARD_H
