#ifndef __AS_CONSUMER_H__
#define __AS_CONSUMER_H__

#include <hc.hpp>
#include <hsa/hsa.h>

#include <iostream>
#define VERBOSE_PRINT(xxx)                                                     \
  do {                                                                         \
    if (verbose_mode) {                                                        \
      xxx;                                                                     \
    }                                                                          \
  } while (false)

#define ATTR_GLOBAL __attribute__((address_space(1)))

const unsigned int THREADS_PER_BLOCK = 123; // include a partial warp
const unsigned int NUM_BLOCKS = 3; // because powers of two are too convenient
const unsigned int NUM_THREADS = NUM_BLOCKS * THREADS_PER_BLOCK;
const unsigned int NUM_PACKETS_INSUFFICIENT = NUM_THREADS - 23;
const unsigned int NUM_PACKETS_LARGE = NUM_THREADS * 4;

bool verbose_mode = false;
bool synchronous_mode = false;

bool validate_options(int argc, char *argv[]) {
  for (int ii = 1; ii != argc; ++ii) {
    char *str = argv[ii];
    if (str[0] != '-')
      return false;
    if (str[2])
      return false;
    switch (str[1]) {
    case 'v':
      verbose_mode = true;
      break;
    case 's':
      synchronous_mode = true;
      break;
    default:
      return false;
      break;
    }
  }

  return true;
}

bool parse_options(int argc, char *argv[]) {
  verbose_mode = false;
  synchronous_mode = false;

  if (!validate_options(argc, argv)) {
    std::cout << "invalid command-line arguments" << std::endl;
    return false;
  }

  return true;
}

unsigned int read_uint(const unsigned char *ptr) {
  unsigned int value = 0;

  for (int ii = sizeof(unsigned int) - 1; ii >= 0; --ii) {
    value <<= 8;
    value |= ptr[ii];
  }

  return value;
}

/* BEGIN DUPLICATION */

// TODO: The host code below is duplicated, and should ideally reside
// in a common host-side library.

#define __OCKL_AS_SIGNAL_INIT ((uint64_t)(-1))
#define __OCKL_AS_SIGNAL_DONE ((uint64_t)(-2))

typedef enum {
  __OCKL_AS_PACKET_EMPTY = 0,
  __OCKL_AS_PACKET_READY = 1
} __ockl_as_packet_type_t;

#define __OCKL_AS_PAYLOAD_ALIGNMENT 4
#define __OCKL_AS_PAYLOAD_BYTES 48

typedef enum {
  __OCKL_AS_PACKET_HEADER_TYPE = 0, // corresponds to HSA_PACKET_HEADER_TYPE
  __OCKL_AS_PACKET_HEADER_RESERVED0 = 8,
  __OCKL_AS_PACKET_HEADER_FLAGS = 13,
  __OCKL_AS_PACKET_HEADER_BYTES = 16,
  __OCKL_AS_PACKET_HEADER_SERVICE = 24,
} __ockl_as_packet_header_t;

typedef enum {
  __OCKL_AS_PACKET_HEADER_WIDTH_TYPE = 8,
  __OCKL_AS_PACKET_HEADER_WIDTH_RESERVED0 = 5,
  __OCKL_AS_PACKET_HEADER_WIDTH_FLAGS = 3,
  __OCKL_AS_PACKET_HEADER_WIDTH_BYTES = 8,
  __OCKL_AS_PACKET_HEADER_WIDTH_SERVICE = 8
} __ockl_as_packet_header_width_t;

// A packet is 64 bytes long, and the payload starts at index 16.
struct __ockl_as_packet_t {
  unsigned int header;
  unsigned int reserved1;
  unsigned long connection_id;

  unsigned char payload[__OCKL_AS_PAYLOAD_BYTES];
};

typedef enum {
  __OCKL_AS_STATUS_SUCCESS,
  __OCKL_AS_STATUS_INVALID_REQUEST,
  __OCKL_AS_STATUS_OUT_OF_RESOURCES,
  __OCKL_AS_STATUS_BUSY,
  __OCKL_AS_STATUS_UNKNOWN_ERROR
} __ockl_as_status_t;

typedef enum {
  __OCKL_AS_CONNECTION_BEGIN = 1,
  __OCKL_AS_CONNECTION_END = 2,
  __OCKL_AS_CONNECTION_FLUSH = 4
} __ockl_as_flag_t;

typedef enum {
  __OCKL_AS_BUILTIN_SERVICE_PRINTF = 42
} __ockl_as_builtin_service_t;

typedef struct {
  // Opaque handle. The value 0 is reserved.
  unsigned long handle;
} __ockl_as_signal_t;

typedef struct __ockl_as_packet_t __ockl_as_packet_t;

typedef struct {
  unsigned long read_index;
  unsigned long write_index;
  __ockl_as_signal_t doorbell_signal;
  __ockl_as_packet_t *base_address;
  unsigned long size;
} __ockl_as_stream_t;

extern "C" __ockl_as_status_t __ockl_as_write_block(
    __ockl_as_stream_t __attribute__((address_space(1))) * stream,
    unsigned char service_id, unsigned long *connection_id,
    const unsigned char *str, uint len, unsigned char flags);

namespace hc {
__ockl_as_status_t as_write_block(__ockl_as_stream_t *stream,
                                  unsigned char service_id,
                                  unsigned long *connection_id,
                                  const unsigned char *str, uint32_t len,
                                  unsigned char flags) [[hc]] {
  auto *gstream = reinterpret_cast<__ockl_as_stream_t  ATTR_GLOBAL*>(stream);
  return __ockl_as_write_block(gstream, service_id, connection_id, str, len,
                               flags);
}
} // namespace hc

uint8_t get_header_field(uint32_t header, uint8_t offset, uint8_t size) {
  return (header >> offset) & ((1 << size) - 1);
}

uint8_t get_packet_type(uint32_t header) {
  return get_header_field(header, __OCKL_AS_PACKET_HEADER_TYPE,
                          __OCKL_AS_PACKET_HEADER_WIDTH_TYPE);
}

uint8_t get_packet_flags(uint32_t header) {
  return get_header_field(header, __OCKL_AS_PACKET_HEADER_FLAGS,
                          __OCKL_AS_PACKET_HEADER_WIDTH_FLAGS);
}

uint8_t get_packet_bytes(uint32_t header) {
  return get_header_field(header, __OCKL_AS_PACKET_HEADER_BYTES,
                          __OCKL_AS_PACKET_HEADER_WIDTH_BYTES);
}

uint8_t get_packet_service(uint32_t header) {
  return get_header_field(header, __OCKL_AS_PACKET_HEADER_SERVICE,
                          __OCKL_AS_PACKET_HEADER_WIDTH_SERVICE);
}

const unsigned int __OCKL_AS_PACKET_SIZE = sizeof(__ockl_as_packet_t);

bool ockl_as_stream_create(hsa_region_t region, uint32_t num_packets,
                           hsa_signal_t doorbell_signal,
                           __ockl_as_stream_t **stream) {
  void *ptr;
  size_t buffer_size =
      sizeof(__ockl_as_stream_t) + num_packets * sizeof(__ockl_as_packet_t);

  hsa_status_t status = hsa_memory_allocate(region, buffer_size, &ptr);
  if (status != HSA_STATUS_SUCCESS)
    return false;

  memset(ptr, 0, buffer_size);
  __ockl_as_stream_t *r = reinterpret_cast<__ockl_as_stream_t *>(ptr);
  r->base_address = reinterpret_cast<__ockl_as_packet_t *>(&r[1]);
  r->size = num_packets;

  if (!synchronous_mode) {
    r->doorbell_signal = {doorbell_signal.handle};
  }
  VERBOSE_PRINT(std::cout << "initial doorbell: " << r->doorbell_signal.handle
                          << std::endl);

  *stream = r;
  return true;
}

/* END DUPLICATION */

hsa_region_t *get_region() {
  hc::accelerator acc;
  if (!acc.is_hsa_accelerator()) {
    return nullptr;
  }

  auto _r = acc.get_hsa_am_finegrained_system_region();
  if (!_r) {
    return nullptr;
  }
  auto *region = reinterpret_cast<hsa_region_t *>(_r);

  return region;
}

__ockl_as_stream_t *createStream(hsa_region_t region,
                                 unsigned int num_packets) {
  hsa_signal_t signal;
  if (hsa_signal_create(UINT64_MAX, 0, NULL, &signal) != HSA_STATUS_SUCCESS) {
    return nullptr;
  }

  __ockl_as_stream_t *stream;
  if (!ockl_as_stream_create(region, num_packets, signal, &stream)) {
    return nullptr;
  }

  return stream;
}

const std::chrono::seconds watchdog_timeout(2);

class ockl_as_consumer_t {
  bool packets_valid;
  bool terminated_successfully;
  uint64_t signal_value;

  std::chrono::steady_clock::time_point watchdog_start;

  void handle_packet(uint8_t service, uint64_t connection_id, uint8_t *payload,
                     uint8_t bytes, uint8_t flags);

  bool wait_on_signal(__ockl_as_signal_t doorbell, uint64_t timeout);
  bool consume_one_packet(uint64_t read_index, uint64_t write_index);
  void consume_packets();
  bool check();
  virtual bool check_packet(uint8_t service, uint64_t connection_id,
                            uint8_t *payload, uint8_t bytes, uint8_t flags) = 0;
  virtual bool check_final() = 0;

public:
  __ockl_as_stream_t *stream;

  ockl_as_consumer_t(__ockl_as_stream_t *r)
      : packets_valid(true), terminated_successfully(false),
        signal_value(__OCKL_AS_SIGNAL_INIT), stream(r){};
  virtual ~ockl_as_consumer_t(){};
  bool run(hc::completion_future cf);
};

void ockl_as_consumer_t::handle_packet(uint8_t service, uint64_t connection_id,
                                       uint8_t *payload, uint8_t bytes,
                                       uint8_t flags) {
  packets_valid &= check_packet(service, connection_id, payload, bytes, flags);
}

bool ockl_as_consumer_t::check() {
  if (!packets_valid)
    return false;
  if (!check_final())
    return false;
  return terminated_successfully;
}

bool ockl_as_consumer_t::consume_one_packet(uint64_t read_index,
                                            uint64_t write_index) {
  VERBOSE_PRINT(std::cout << "read index: " << read_index
                          << "\twrite index: " << write_index << std::endl);

  if (read_index == write_index) {
    VERBOSE_PRINT(std::cout << "no new packet" << std::endl);
    return false;
  }

  auto *packets = stream->base_address;
  auto *packet = packets + read_index % stream->size;

  auto header = __atomic_load_n(&packet->header, std::memory_order_acquire);
  auto type = get_packet_type(header);

  VERBOSE_PRINT(std::cout << "packet address: " << packet << std::endl);
  VERBOSE_PRINT(std::cout << "packet header: " << header << std::endl);
  VERBOSE_PRINT(std::cout << "packet type: " << (uint16_t)type << std::endl);

  VERBOSE_PRINT(std::cout << "expected type: "
                          << (uint16_t)__OCKL_AS_PACKET_READY << std::endl);

  if (type != __OCKL_AS_PACKET_READY) {
    return false;
  }

  VERBOSE_PRINT(std::cout << "processing packet" << std::endl);

  auto service = get_packet_service(header);
  auto connection_id = packet->connection_id;
  auto flags = get_packet_flags(header);
  auto bytes = get_packet_bytes(header);
  handle_packet(service, connection_id, packet->payload, bytes, flags);
  __atomic_store_n(&packet->header, 0, std::memory_order_release);

  return true;
}

bool ockl_as_consumer_t::wait_on_signal(__ockl_as_signal_t ockl_doorbell,
                                        uint64_t timeout) {
  hsa_signal_t doorbell = {ockl_doorbell.handle};
  VERBOSE_PRINT(std::cout << "signal value: " << (int64_t)signal_value
                          << std::endl);

  do {
    auto now = std::chrono::steady_clock::now();
    if (now - watchdog_start > watchdog_timeout) {
      std::cout << "watchdog timed out on signal" << std::endl;
      return false;
    }

    if (signal_value == __OCKL_AS_SIGNAL_DONE) {
      VERBOSE_PRINT(std::cout << "terminating asynchronous operation"
                              << std::endl);
      terminated_successfully = true;
      return false;
    }

    auto new_signal_value = hsa_signal_wait_scacquire(
        doorbell, HSA_SIGNAL_CONDITION_NE, signal_value, timeout,
        HSA_WAIT_STATE_BLOCKED);
    VERBOSE_PRINT(std::cout << "new signal value: " << new_signal_value
                            << std::endl);
    if (new_signal_value != signal_value) {
      signal_value = new_signal_value;
      return true;
    }
  } while (true);
  return false;
}

void ockl_as_consumer_t::consume_packets() {
  VERBOSE_PRINT(std::cout << "launched consumer" << std::endl);
  VERBOSE_PRINT(std::cout << "asynchronous mode: "
                          << (stream->doorbell_signal.handle != 0
                                  ? "enabled"
                                  : "disabled")
                          << std::endl);
  VERBOSE_PRINT(std::cout << "packet base: " << stream->base_address
                          << std::endl);
  VERBOSE_PRINT(std::cout << "num packets: " << stream->size << std::endl);
  VERBOSE_PRINT(std::cout << "doorbell: " << stream->doorbell_signal.handle
                          << std::endl);

  auto read_index =
      __atomic_load_n(&stream->read_index, std::memory_order_acquire);
  assert(read_index == 0);

  auto doorbell = stream->doorbell_signal;
  watchdog_start = std::chrono::steady_clock::now();

  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (now - watchdog_start > watchdog_timeout) {
      std::cout << "watchdog timed out in main loop" << std::endl;
      break;
    }

    auto write_index =
        __atomic_load_n(&stream->write_index, std::memory_order_acquire);

    if (consume_one_packet(read_index, write_index)) {
      ++read_index;
      __atomic_store_n(&stream->read_index, read_index,
                       std::memory_order_release);
      continue;
    }

    if (synchronous_mode) {
      terminated_successfully = true;
      VERBOSE_PRINT(std::cout << "terminating synchronous operations"
                              << std::endl);
      break;
    }

    // If we have run out of work to do, check if main wants us to exit.
    uint64_t timeout = UINT64_MAX;
    if (read_index == write_index) {
      // Either there is no more work to do, or work is simply not
      // visible yet. These two situations are indistinguishable, so
      // we temporarily reduce the signal timeout.
      timeout = 1024 * 1024;
    }

    if (!wait_on_signal(doorbell, timeout)) {
      break;
    }
  }
}

bool ockl_as_consumer_t::run(hc::completion_future cf) {
  if (synchronous_mode) {
    /* FIXME: We would like this to work, but it doesn't
    if (cf.wait_for(std::chrono::seconds(1)) != std::future_status::ready) {
        std::cout << "pfe timed out" << std::endl;
        return false;
    }

       Instead we use a poor man's timeout for the PFE:
    */
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (!cf.is_ready()) {
      VERBOSE_PRINT(std::cout << "pfe timed out" << std::endl);
      abort();
    }
  } else {
    // In asynchronous mode, spawn a thread that waits for the
    // future to complete, and then signals the consumer.
    cf.then([=]() {
      hsa_signal_t signal = {stream->doorbell_signal.handle};
      hsa_signal_store_screlease(signal, __OCKL_AS_SIGNAL_DONE);
    });
  }
  consume_packets();
  if (check())
    return true;

  return false;
}

#endif
