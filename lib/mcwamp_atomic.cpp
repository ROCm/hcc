#include <mutex>
#include <algorithm>

// FIXME : need to consider how to let hc namespace could also use functions here
namespace Concurrency {

std::mutex afa_u, afa_i, afa_f;
unsigned int atomic_add_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afa_u);
    *x += y;
    return *x;
}
int atomic_add_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afa_i);
    *x += y;
    return *x;
}
float atomic_add_float(float* x, float y) {
    std::lock_guard<std::mutex> guard(afa_f);
    *x += y;
    return *x;
}

std::mutex afs_u, afs_i, afs_f;
unsigned int atomic_sub_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afa_u);
    *x -= y;
    return *x;
}
int atomic_sub_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afa_i);
    *x -= y;
    return *x;
}
float atomic_sub_float(float* x, float y) {
    std::lock_guard<std::mutex> guard(afa_f);
    *x -= y;
    return *x;
}

std::mutex afm_u, afm_i;
unsigned int atomic_max_unsigned(unsigned int *p, unsigned int val) {
    std::lock_guard<std::mutex> guard(afm_u);
    *p = std::max(*p, val);
    return *p;
}
int atomic_max_int(int *p, int val) {
    std::lock_guard<std::mutex> guard(afm_i);
    *p = std::max(*p, val);
    return *p;
}

std::mutex afi_u, afi_i;
unsigned int atomic_inc_unsigned(unsigned int *p) {
    std::lock_guard<std::mutex> guard(afi_u);
    *p += 1;
    return *p;
}
int atomic_inc_int(int *p) {
    std::lock_guard<std::mutex> guard(afi_i);
    *p += 1;
    return *p;
}

}
