#ifndef INCLUDE_AMP_IMPL_H
#define INCLUDE_AMP_IMPL_H
// Specialization of AMP classes/templates

namespace Concurrency {

// 1-D Concurrency::index specialization
template<>
class index<1> {
 public:
  explicit index(int i0) restrict(amp,cpu):index_(i0) {}
  int operator[](unsigned int c) const restrict(amp,cpu) { return index_; }
 private:
  int index_;
};
} //namespace Concurrency
#endif //INCLUDE_AMP_IMPL_H
