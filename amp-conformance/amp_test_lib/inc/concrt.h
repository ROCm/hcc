#include <stddef.h>

namespace Concurrency {

const unsigned int COOPERATIVE_TIMEOUT_INFINITE = (unsigned int)-1;

class event {
public:
  event() {}
  ~event() {}
  size_t wait(unsigned int _Timeout = COOPERATIVE_TIMEOUT_INFINITE) { return 0;}
  void set() {}
  void reset() {}

private:
  event(const event& _Event);
  event& operator=(const event& _Event);
  static const unsigned int timeout_infinite = COOPERATIVE_TIMEOUT_INFINITE;
};

}//namespace Concurrency
