#include <mutex>
#include <algorithm>

// FIXME : need to consider how to let hc namespace could also use functions here
namespace Concurrency {

std::mutex afx_u, afx_i, afx_f;
unsigned int atomic_exchange_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afx_u);
    unsigned int old = *x;
    *x = y;
    return old;
}
int atomic_exchange_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afx_i);
    int old = *x;
    *x = y;
    return old;
}
float atomic_exchange_float(float* x, float y) {
    std::lock_guard<std::mutex> guard(afx_f);
    int old = *x;
    *x = y;
    return old;
}

std::mutex afcas_u, afcas_i;
unsigned int atomic_compare_exchange_unsigned(unsigned int *x, unsigned int y, unsigned int z) {
    std::lock_guard<std::mutex> guard(afcas_u);
    unsigned int old = *x;
    if (*x == y) {
        *x = z;
    }
    return old;
}
int atomic_compare_exchange_int(int *x, int y, int z) {
    std::lock_guard<std::mutex> guard(afcas_i);
    int old = *x;
    if (*x == y) {
        *x = z;
    }
    return old;
}

std::mutex afa_u, afa_i, afa_f;
unsigned int atomic_add_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afa_u);
    unsigned int old = *x;
    *x += y;
    return old;
}
int atomic_add_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afa_i);
    int old = *x;
    *x += y;
    return old;
}
float atomic_add_float(float* x, float y) {
    std::lock_guard<std::mutex> guard(afa_f);
    float old = *x;
    *x += y;
    return old;
}

std::mutex afs_u, afs_i, afs_f;
unsigned int atomic_sub_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afa_u);
    unsigned int old = *x;
    *x -= y;
    return old;
}
int atomic_sub_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afa_i);
    int old = *x;
    *x -= y;
    return old;
}
float atomic_sub_float(float* x, float y) {
    std::lock_guard<std::mutex> guard(afa_f);
    float old = *x;
    *x -= y;
    return old;
}

std::mutex afand_u, afand_i;
unsigned int atomic_and_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afand_u);
    unsigned int old = *x;
    *x &= y;
    return old;
}
int atomic_and_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afand_i);
    int old = *x;
    *x &= y;
    return old;
}

std::mutex afor_u, afor_i;
unsigned int atomic_or_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afor_u);
    unsigned int old = *x;
    *x |= y;
    return old;
}
int atomic_or_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afor_i);
    int old = *x;
    *x |= y;
    return old;
}

std::mutex afxor_u, afxor_i;
unsigned int atomic_xor_unsigned(unsigned int *x, unsigned int y) {
    std::lock_guard<std::mutex> guard(afxor_u);
    unsigned int old = *x;
    *x ^= y;
    return old;
}
int atomic_xor_int(int *x, int y) {
    std::lock_guard<std::mutex> guard(afxor_i);
    int old = *x;
    *x ^= y;
    return old;
}

std::mutex afmax_u, afmax_i;
unsigned int atomic_max_unsigned(unsigned int *p, unsigned int val) {
    std::lock_guard<std::mutex> guard(afmax_u);
    unsigned int old = *p;
    *p = std::max(*p, val);
    return old;
}
int atomic_max_int(int *p, int val) {
    std::lock_guard<std::mutex> guard(afmax_i);
    int old = *p;
    *p = std::max(*p, val);
    return old;
}

std::mutex afmin_u, afmin_i;
unsigned int atomic_min_unsigned(unsigned int *p, unsigned int val) {
    std::lock_guard<std::mutex> guard(afmin_u);
    unsigned int old = *p;
    *p = std::min(*p, val);
    return old;
}
int atomic_min_int(int *p, int val) {
    std::lock_guard<std::mutex> guard(afmin_i);
    int old = *p;
    *p = std::min(*p, val);
    return old;
}

std::mutex afi_u, afi_i;
unsigned int atomic_inc_unsigned(unsigned int *p) {
    std::lock_guard<std::mutex> guard(afi_u);
    unsigned int old = *p;
    *p += 1;
    return old;
}
int atomic_inc_int(int *p) {
    std::lock_guard<std::mutex> guard(afi_i);
    int old = *p;
    *p += 1;
    return old;
}

std::mutex afd_u, afd_i;
unsigned int atomic_dec_unsigned(unsigned int *p) {
    std::lock_guard<std::mutex> guard(afd_u);
    unsigned int old = *p;
    *p -= 1;
    return old;
}
int atomic_dec_int(int *p) {
    std::lock_guard<std::mutex> guard(afd_i);
    int old = *p;
    *p -= 1;
    return old;
}

}
