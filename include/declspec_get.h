#pragma once
// Mimic common use cases of __declspec(get) for `extent' virtual
// data member in array/array_views
template <class T, typename E>
class DeclSpecGetExtent {
 public:
  DeclSpecGetExtent (T *t) restrict(amp,cpu):
    t_(reinterpret_cast<intptr_t>(t)) {}

  operator E() const restrict(amp,cpu) { //cast to T
    return reinterpret_cast<T*>(t_)->get_extent();
  }

  bool operator==(const T &t) const restrict(amp,cpu){
    return reinterpret_cast<T*>(t_)->get_extent() == t;
  }

  // Forwording member calls to Concurrency::extent
  int size() const restrict(amp,cpu) {
    return this->operator E().size();
  }

  int operator[](unsigned int c) const restrict(amp, cpu) {
    return this->operator E().operator[](c);
  }
  
  template <int D0>
  tiled_extent<D0> tile() const {
    T* t= reinterpret_cast<T*>(t_);
    extent<1> e = t->get_extent();
    return e.tile<D0>();
  }

  template <int D0, int D1>
  tiled_extent<D0, D1> tile() const {
    T* t= reinterpret_cast<T*>(t_);
    extent<2> e = t->get_extent();
    return e.tile<D0, D1>();
  }

  __attribute__((annotate("serialize")))
  void __cxxamp_serialize(Serialize& s) const {};

  __attribute__((annotate("deserialize")))
  DeclSpecGetExtent(void) restrict(amp) {};
 private:
  intptr_t t_;
};

