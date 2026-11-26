#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
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
 * Consume each accepted handle once with Wait. Source and target memory must
 * remain valid until Wait returns. Operation outputs are supplied to Wait.
 */
class ExternalStoreBackend {
   public:
    using TaskHandle = void*;

    struct TaskOutput {
        // Values are in input key order and valid only for successful items.
        std::vector<uint8_t> exists;
        std::vector<size_t> actual_sizes;
    };

    virtual ~ExternalStoreBackend() = default;

    /**
     * @brief Initializes the backend.
     * @param options Backend-specific initialization options.
     * @param tenant_id Tenant bound to this backend instance.
     * @return Success or an initialization ErrorCode.
     */
    virtual tl::expected<void, ErrorCode> Init(
        const std::unordered_map<std::string, std::string>& options,
        const TenantId& tenant_id) = 0;

    /**
     * @brief Drains tasks and releases backend resources, including registered
     * memory. Call after client operations stop; repeated shutdown is safe.
     * @return Success or a shutdown ErrorCode.
     */
    virtual tl::expected<void, ErrorCode> Shutdown() = 0;

    /**
     * @brief Submits complete-object writes in slice order.
     * @param keys Object keys to write.
     * @param sources Source slices per key, in object-byte order.
     * Source memory stays valid and unchanged until Wait returns.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchPutAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& sources) = 0;

    /**
     * @brief Submits writes at the specified object-byte offsets.
     * @param keys Object keys to write.
     * @param sources Source slices per key; lifetimes follow BatchPutAsync.
     * @param target_offsets Object-byte offsets per source slice.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchPutRangesAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& sources,
        const std::vector<std::vector<size_t>>& target_offsets) = 0;

    /**
     * @brief Submits object deletion independently of native replica leases.
     * @param keys Object keys to remove.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchRemoveAsync(
        const std::vector<ObjectKey>& keys) = 0;

    /**
     * @brief Submits object existence queries. Wait reports 0 (missing) or 1
     * (present) per key in exists.
     * @param keys Object keys to query.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchExistsAsync(
        const std::vector<ObjectKey>& keys) = 0;

    /**
     * @brief Submits complete-object reads. Wait reports complete object sizes
     * in actual_sizes, in input key order.
     * @param keys Object keys to read.
     * @param targets Destination slices per key, in object-byte order. Sizes
     * are writable capacities.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchGetAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets) = 0;

    /**
     * @brief Submits exact byte-range reads into the corresponding slices.
     * Wait reports bytes read in actual_sizes, in input key order.
     * @param keys Object keys to read.
     * @param targets Destination slices per key; sizes are exact read lengths.
     * @param source_offsets Object-byte offsets matching each target slice.
     * @return Accepted task handle or a submission ErrorCode.
     */
    virtual tl::expected<TaskHandle, ErrorCode> BatchGetRangesAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const std::vector<std::vector<size_t>>& source_offsets) = 0;

    /**
     * @brief Waits for completion and consumes the task handle. All task access
     * to memory and registrations ends before return, including on timeout.
     * @param task Handle consumed on entry, including if Wait throws.
     * @param output Operation results, valid only for successful items. Fields
     * not used by the operation are empty. Must remain valid during this call.
     * @return One completion status per input key, in input order. Operation
     * outputs are valid only for successful items.
     */
    virtual std::vector<tl::expected<void, ErrorCode>> Wait(
        TaskHandle task, TaskOutput& output) = 0;

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

/**
 * @brief Owns an accepted backend task and waits on destruction if still
 * pending.
 */
class ExternalStoreTaskGuard {
   public:
    ExternalStoreTaskGuard() = default;

    /**
     * @brief Takes ownership of an accepted task without owning its backend.
     * @param backend Backend that stays usable until the task is consumed.
     * @param task Handle consumed by Wait or destruction.
     */
    ExternalStoreTaskGuard(ExternalStoreBackend* backend,
                           ExternalStoreBackend::TaskHandle task);
    ~ExternalStoreTaskGuard();

    ExternalStoreTaskGuard(const ExternalStoreTaskGuard&) = delete;
    ExternalStoreTaskGuard& operator=(const ExternalStoreTaskGuard&) = delete;
    ExternalStoreTaskGuard(ExternalStoreTaskGuard&& other) noexcept;
    ExternalStoreTaskGuard& operator=(ExternalStoreTaskGuard&& other);

    explicit operator bool() const { return task_.has_value(); }

    /**
     * @brief Waits for completion and consumes the task.
     * @param output Operation results supplied to the backend Wait.
     * @return Backend statuses, or empty statuses and output for an empty or
     * moved-from guard. Such guards perform no backend work.
     */
    std::vector<tl::expected<void, ErrorCode>> Wait(
        ExternalStoreBackend::TaskOutput& output);

   private:
    ExternalStoreBackend* backend_ = nullptr;
    std::optional<ExternalStoreBackend::TaskHandle> task_;
};

}  // namespace mooncake
