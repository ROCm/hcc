// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out && %t.out -s

#include "as_consumer.h"
#include <unordered_map>

#define STR30 "Cras nec volutpat mi, sed sed."
#define STR47 "Lorem ipsum dolor sit amet, consectetur nullam."
#define STR60 "Curabitur id maximus nibh. Donec quis porttitor nisl nullam."
#define STR95                                                                  \
  "In mollis imperdiet nibh nec ullamcorper."                                  \
  " Suspendisse placerat massa iaculis ipsum viverra sed."
#define STR124                                                                 \
  "Proin ut diam sit amet erat mollis gravida ac non sem."                     \
  " Mauris viverra leo metus, id luctus metus feugiat sed. Morbi "             \
  "posuere."

#define DECLARE_TEST_DATA()                                                    \
  const char *str30 = STR30;                                                   \
  const char *str60 = STR60;                                                   \
  const char *str47 = STR47;                                                   \
  const char *str95 = STR95;                                                   \
  const char *str124 = STR124;                                                 \
  const int numStr = 5;                                                        \
  const char *strArray[5] = {str30, str60, str47, str95, str124};              \
  unsigned char strLengths[5] = {30, 60, 47, 95, 124};

class test_handler_t : public ockl_as_consumer_t {
  typedef std::unordered_map<uint64_t, std::string> stream_buffer_map_t;
  stream_buffer_map_t buffers;
  std::unordered_map<std::string, int> strRecd;

  DECLARE_TEST_DATA();

  bool check_packet(uint8_t service, uint64_t connection_id, uint8_t *payload,
                    uint8_t bytes, uint8_t flags) {
    if ((flags & __OCKL_AS_CONNECTION_BEGIN) !=
        (buffers.count(connection_id) == 0))
      return false;

    std::string &buf = buffers[connection_id];
    buf.insert(buf.end(), payload, payload + bytes);

    if (flags & __OCKL_AS_CONNECTION_END) {
      if (buf != strArray[service])
        return false;
      strRecd[buf] += 1;
      buffers.erase(connection_id);
    }

    return true;
  }

  bool check_final() {
    VERBOSE_PRINT(std::cout << "expected counts:");
    int expected_counts[numStr];
    for (int ii = 0; ii != numStr; ++ii) {
      expected_counts[ii] = NUM_THREADS / numStr;
      if (ii < (NUM_THREADS % numStr)) {
        ++expected_counts[ii];
      }
      VERBOSE_PRINT(std::cout << " " << expected_counts[ii]);
    }
    VERBOSE_PRINT(std::cout << std::endl);

    if (strRecd.size() != numStr)
      return false;

    for (int ii = 0; ii != numStr; ++ii) {
      std::string mystr(strArray[ii]);
      VERBOSE_PRINT(std::cout << mystr << ": " << strRecd[mystr] << std::endl);
      if (strRecd[mystr] != expected_counts[ii])
        return false;
    }

    return true;
  }

public:
  test_handler_t(__ockl_as_stream_t *stream) : ockl_as_consumer_t(stream){};
};

void mixedProducers(__ockl_as_stream_t *stream, unsigned int tid) [[hc]] {
  DECLARE_TEST_DATA();
  const unsigned int idx = tid % 5;

  uint64_t connection_id;

  hc::as_write_block(stream, idx, &connection_id,
                     reinterpret_cast<unsigned const char *>(strArray[idx]),
                     strLengths[idx],
                     __OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END);
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
        mixedProducers(stream, idx.global[0]);
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
