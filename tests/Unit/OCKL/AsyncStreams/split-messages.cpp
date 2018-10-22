// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out && %t.out -s

#include "as_consumer.h"
#include <unordered_map>

#define STR27 "In et consectetur mi metus."
#define STR64 "Praesent tempus arcu id ligula blandit, eget congue justo metus."
#define STR40 "Sed at dolor ipsum. Curabitur cras amet."

class test_handler_t : public ockl_as_consumer_t {
  typedef std::unordered_map<uint64_t, std::string> stream_buffer_map_t;
  stream_buffer_map_t buffers;
  std::unordered_map<std::string, int> strRecd;

  static const int numExpected = 3;
  const char *strExpected[numExpected] = {STR27 STR64 STR40, STR64 STR40 STR27,
                                          STR40 STR27 STR64};

  bool check_packet(uint8_t service, uint64_t connection_id, uint8_t *payload,
                    uint8_t bytes, uint8_t flags) {
    if ((flags & __OCKL_AS_CONNECTION_BEGIN) !=
        (buffers.count(connection_id) == 0))
      return false;

    std::string &buf = buffers[connection_id];
    buf.insert(buf.end(), payload, payload + bytes);

    if (flags & __OCKL_AS_CONNECTION_END) {
      if (buf != strExpected[service])
        return false;
      strRecd[buf] += 1;
      buffers.erase(connection_id);
    }

    return true;
  }

  bool check_final() {
    VERBOSE_PRINT(std::cout << "expected counts:");
    int expected_counts[numExpected];
    for (int ii = 0; ii != numExpected; ++ii) {
      expected_counts[ii] = NUM_THREADS / numExpected;
      if (ii < (NUM_THREADS % numExpected)) {
        ++expected_counts[ii];
      }
      VERBOSE_PRINT(std::cout << " " << expected_counts[ii]);
    }
    VERBOSE_PRINT(std::cout << std::endl);

    if (strRecd.size() != numExpected)
      return false;

    for (int ii = 0; ii != numExpected; ++ii) {
      std::string mystr(strExpected[ii]);
      VERBOSE_PRINT(std::cout << mystr << ": " << strRecd[mystr] << std::endl);
      if (strRecd[mystr] != expected_counts[ii])
        return false;
    }

    return true;
  }

public:
  test_handler_t(__ockl_as_stream_t *stream) : ockl_as_consumer_t(stream){};
};

void splitMessages(__ockl_as_stream_t *stream, unsigned int tid) [[hc]] {
  const char *str27 = STR27;
  const char *str64 = STR64;
  const char *str40 = STR40;
  const int numStr = 3;
  const char *strArray[] = {str27, str64, str40};
  int strLengths[] = {27, 64, 40};

  int service = tid % 3;
  int first = tid % 3;
  int second = (tid + 1) % 3;
  int third = (tid + 2) % 3;

  uint64_t connection_id;
  hc::as_write_block(stream, service, &connection_id,
                     reinterpret_cast<unsigned const char *>(strArray[first]),
                     strLengths[first],
                     __OCKL_AS_CONNECTION_BEGIN);
  hc::as_write_block(stream, service, &connection_id,
                     reinterpret_cast<unsigned const char *>(strArray[second]),
                     strLengths[second], 0);
  hc::as_write_block(stream, service, &connection_id,
                     reinterpret_cast<unsigned const char *>(strArray[third]),
                     strLengths[third],
                     __OCKL_AS_CONNECTION_END);
}

bool run_test() {
  unsigned int num_packets =
      (synchronous_mode ? NUM_PACKETS_LARGE : NUM_THREADS / 7);

  hsa_region_t *region = get_region();
  if (!region)
    return false;

  __ockl_as_stream_t *stream = createStream(*region, num_packets);
  if (!stream)
    return false;

  test_handler_t test_handler(stream);

  hc::extent<1> ex(NUM_THREADS);
  hc::tiled_extent<1> et(ex.tile(THREADS_PER_BLOCK));

  hc::completion_future cf =
    parallel_for_each(et, [=](hc::tiled_index<1> & idx) [[hc]] {
      splitMessages(stream, idx.global[0]);
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
