// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out && %t.out -s

#include "as_consumer.h"

#define STR_HELLO_WORLD "hello world"
#define STRLEN_HELLO_WORLD 11

class test_handler_t : public ockl_as_consumer_t {
  bool check_packet(uint8_t service, uint64_t connection_id, uint8_t *payload,
                    uint8_t bytes, uint8_t flags) {
    if (service != 42)
      return false;
    if (bytes != STRLEN_HELLO_WORLD)
      return false;
    if (flags != (__OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END))
      return false;
    if (0 != strcmp(STR_HELLO_WORLD, reinterpret_cast<const char *>(payload)))
      return false;
    return true;
  }

  bool check_final() { return stream->write_index == 1; }

public:
  test_handler_t(__ockl_as_stream_t *stream) : ockl_as_consumer_t(stream){};
};

static void singlePacketSingleProducer(__ockl_as_stream_t *stream) [[hc]] {
  uint64_t connection_id;

  hc::as_write_block(stream, 42, &connection_id,
                     reinterpret_cast<const uint8_t *>(STR_HELLO_WORLD),
                     STRLEN_HELLO_WORLD,
                     __OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END);
}

bool run_test() {
  unsigned int numPackets = 1;

  hsa_region_t *region = get_region();
  if (!region)
    return false;

  __ockl_as_stream_t *stream = createStream(*region, numPackets);
  if (!stream)
    return false;

  test_handler_t test_handler(stream);

  hc::extent<1> ex(1);
  hc::completion_future cf =
    parallel_for_each(ex, [=](hc::index<1> & idx) [[hc]] {
      singlePacketSingleProducer(stream);
    });

  return test_handler.run(cf);
}

int main(int argc, char **argv) {
  if (!parse_options(argc, argv))
    return -1;
  if (run_test())
    return 0;
  return 1;
}
