#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <ylt/util/tl/expected.hpp>

#include "types.h"

namespace mooncake {

/**
 * @brief Keyed storage accessed independently of Master replicas.
 *
 * Async submissions return a task handle on acceptance, not completion.
 * Consume each accepted handle once with Wait. Source and target memory and
 * output containers must remain valid until Wait returns; callers must not
 * access output containers while work is pending.
 */
class ExternalStoreBackend {
   public:
    using TaskHandle = void*;

    virtual ~ExternalStoreBackend() = default;

    /**
     * @brief Initializes the backend.
     * @param options Backend-specific initialization options.
     * @return Success or an initialization ErrorCode.
     */
    virtual tl::expected<void, ErrorCode> Init(
        const std::unordered_map<std::string, std::string>& options) = 0;

    /**
     * @brief Drains tasks and releases backend resources, including registered
     * memory. Call after client operations stop; repeated shutdown is safe.
     * @return Success or a shutdown ErrorCode.
     */
    virtual tl::expected<void, ErrorCode> Shutdown() = 0;

    /**
     * @brief Submits object existence queries.
     * @param keys Object keys to query.
     * @param tenant_id Tenant for the operation.
     * @param exists Non-null output, one entry per key. Successful items report
     * 0 (missing) or 1 (present) after Wait.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchExistsAsync(
        const std::vector<ObjectKey>& keys, const TenantId& tenant_id,
        std::vector<uint8_t>* exists) = 0;

    /**
     * @brief Submits complete-object reads.
     * @param keys Object keys to read.
     * @param targets Destination slices per key, in object-byte order. Sizes
     * are writable capacities.
     * @param tenant_id Tenant for the operation.
     * @param actual_sizes Non-null output of complete object sizes for
     * successful items after Wait, in input key order.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchGetAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const TenantId& tenant_id, std::vector<size_t>* actual_sizes) = 0;

    /**
     * @brief Submits exact byte-range reads into the corresponding slices.
     * @param keys Object keys to read.
     * @param targets Destination slices per key; sizes are exact read lengths.
     * @param source_offsets Object-byte offsets matching each target slice.
     * @param tenant_id Tenant for the operation.
     * @param actual_sizes Non-null output of bytes read for successful items
     * after Wait, in input key order.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchGetRangesAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const std::vector<std::vector<size_t>>& source_offsets,
        const TenantId& tenant_id, std::vector<size_t>* actual_sizes) = 0;

    /**
     * @brief Waits for completion and consumes the task handle. All task access
     * to memory and registrations ends before return, including on timeout.
     * @param task Handle from an accepted submission; invalid after this call.
     * @return One completion status per input key, in input order. Operation
     * outputs are valid only for successful items.
     */
    virtual std::vector<tl::expected<void, ErrorCode>> Wait(
        TaskHandle task) = 0;

    /**
     * @brief Registers a transfer region. Unsupported memory is skipped.
     * @param addr Region start address.
     * @param length Region length in bytes.
     * @param location Memory location or topology hint.
     * @return Success if registered or skipped, otherwise an ErrorCode.
     */
    virtual tl::expected<void, ErrorCode> RegisterMemory(
        void* addr, size_t length, const std::string& location) = 0;

    /**
     * @brief Unregisters a region. Keep it alive until this call succeeds.
     * @param addr Start address supplied to RegisterMemory.
     * @return Success or an unregistration ErrorCode.
     */
    virtual tl::expected<void, ErrorCode> UnregisterMemory(void* addr) = 0;
};

}  // namespace mooncake
