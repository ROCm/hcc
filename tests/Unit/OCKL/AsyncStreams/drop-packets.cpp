// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out -s

#include "as_consumer.h"

class test_handler_t : public ockl_as_consumer_t {
  unsigned int *status;

  bool check_packet(uint8_t service, uint64_t connection_id, uint8_t *payload,
                    uint8_t bytes, uint8_t flags) {
    if (service != 42)
      return false;
    if (bytes != sizeof(unsigned int))
      return false;
    if (flags != (__OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END))
      return false;
    unsigned int pp = read_uint(payload);
    if (pp >= NUM_THREADS)
      return false;
    return true;
  }

  bool check_final() {
    if (stream->write_index != NUM_PACKETS_INSUFFICIENT)
      return false;

    unsigned int errorsExpected = NUM_THREADS - NUM_PACKETS_INSUFFICIENT;
    for (int ii = 0; ii != NUM_THREADS; ++ii) {
      switch (status[ii]) {
      case __OCKL_AS_STATUS_OUT_OF_RESOURCES:
        if (errorsExpected == 0)
          return false;
        --errorsExpected;
        break;
      case __OCKL_AS_STATUS_SUCCESS:
        break;
      default:
        return false;
      }
    }
    if (errorsExpected != 0)
      return false;

    return true;
  }

public:
  test_handler_t(__ockl_as_stream_t *stream, unsigned int *s)
      : ockl_as_consumer_t(stream), status(s){};
};

void dropPackets(__ockl_as_stream_t *stream, unsigned int *status,
                 unsigned int tid) [[hc]] {
  uint64_t connection_id;

  status[tid] =
      hc::as_write_block(stream, 42, &connection_id,
                         reinterpret_cast<const unsigned char *>(&tid),
                         sizeof(unsigned int),
                         __OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END);
}

bool run_test() {
  bool success = true;

  hsa_region_t *region = get_region();
  if (!region)
    return false;

  __ockl_as_stream_t *stream = createStream(*region, NUM_PACKETS_INSUFFICIENT);
  if (!stream)
    return false;

  void *status;
  if (hsa_memory_allocate(*region, NUM_THREADS * sizeof(unsigned int),
                          &status) != HSA_STATUS_SUCCESS)
    return false;

  test_handler_t test_handler(stream, reinterpret_cast<unsigned int *>(status));

  hc::extent<1> ex(NUM_THREADS);
  hc::tiled_extent<1> et(ex.tile(THREADS_PER_BLOCK));

  hc::completion_future cf =
      parallel_for_each(et, [=](hc::tiled_index<1> & idx) [[hc]] {
        dropPackets(stream, reinterpret_cast<unsigned int *>(status), idx.global[0]);
      });

  return test_handler.run(cf);
}

int main(int argc, char **argv) {
  if (!parse_options(argc, argv))
    return -1;
  if (!synchronous_mode) {
    std::cout << "asynchronous mode not supported for this test" << std::endl;
    return -2;
  }
  if (run_test())
    return 0;
  return 1;
}
