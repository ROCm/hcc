// RUN: %hc %s -I%hsa_header_path -L%hsa_library_path -lhsa-runtime64 -o %t.out && %t.out && %t.out -s

#include "as_consumer.h"

#define NUM_SERVICES 7

class test_handler_t : public ockl_as_consumer_t {
    int data[NUM_SERVICES] = { 0, };

    bool check_packet(uint8_t service, uint64_t connection_id,
                       uint8_t *payload, uint8_t bytes,
                       uint8_t flags)
    {
        if (flags != (__OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END))
            return false;
        if (bytes != sizeof(unsigned int)) return false;
        unsigned int tid = read_uint(payload);
        if (service != tid % NUM_SERVICES) return false;
        data[service]++;
        return true;
    }

    bool check_final() {
        VERBOSE_PRINT(std::cout << "expected counts:");
        int expected[NUM_SERVICES];
        for (int ii = 0; ii != NUM_SERVICES; ++ii) {
            expected[ii] = NUM_THREADS / NUM_SERVICES;
            if (ii < NUM_THREADS % NUM_SERVICES) {
                expected[ii]++;
            }
            VERBOSE_PRINT(std::cout << " " << expected[ii]);
        }
        VERBOSE_PRINT(std::cout << std::endl);

        if (stream->write_index != NUM_THREADS) return false;

        for (int ii = 0; ii != NUM_SERVICES; ++ii) {
            VERBOSE_PRINT(std::cout << "service " << ii << ": " << data[ii] << std::endl);
            if (data[ii] != expected[ii]) return false;
        }

        return true;
    }

public:
    test_handler_t(__ockl_as_stream_t *stream) : ockl_as_consumer_t(stream) {};
};

void multipleProducers(__ockl_as_stream_t* stream, unsigned int tid) [[hc]]
{
    uint64_t connection_id;
    uint8_t service = tid % NUM_SERVICES;

    hc::as_write_block(stream, service, &connection_id,
                       reinterpret_cast<const unsigned char*>(&tid),
                       sizeof(unsigned int),
                       __OCKL_AS_CONNECTION_BEGIN | __OCKL_AS_CONNECTION_END);
}

bool run_test()
{
    unsigned int num_packets = (synchronous_mode ? NUM_THREADS : 3);

    hsa_region_t *region = get_region();
    if (!region) return false;

    __ockl_as_stream_t *stream = createStream(*region, num_packets);
    if (!stream) return false;

    test_handler_t test_handler(stream);

    hc::extent<1> ex(NUM_THREADS);
    hc::tiled_extent<1> et(ex.tile(THREADS_PER_BLOCK));

    hc::completion_future cf =
      parallel_for_each(et, [=](hc::tiled_index<1> &idx) [[hc]] {
        multipleProducers(stream, idx.global[0]);
      });

    return test_handler.run(cf);
}

int main(int argc, char** argv)
{
    if (!parse_options(argc, argv)) return -1;
    if (run_test()) return 0;
    return 1;
}
