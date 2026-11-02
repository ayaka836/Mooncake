#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>
#include <ylt/util/tl/expected.hpp>

#include "types.h"

namespace mooncake {

// Keyed storage accessed independently of Master replicas.
class ExternalStoreBackend {
   public:
    using TaskHandle = void*;

    virtual ~ExternalStoreBackend() = default;
    virtual tl::expected<void, ErrorCode> Init(
        const std::unordered_map<std::string, std::string>& options) = 0;

    // Call after client operations stop. Drain tasks and release backend
    // resources, including registered memory. Repeated shutdown is safe.
    virtual tl::expected<void, ErrorCode> Shutdown() = 0;
    // Source memory must remain valid and unchanged until Wait returns.
    // Slices describe one object in input order.
    virtual tl::expected<TaskHandle, ErrorCode> PutAsync(
        const ObjectKey& key, const std::vector<Slice>& slices,
        const TenantId& tenant_id) = 0;
    // batched_slices[i] belongs to keys[i]; Wait preserves input key order.
    virtual tl::expected<TaskHandle, ErrorCode> BatchPutAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& batched_slices,
        const TenantId& tenant_id) = 0;
    // Remove the external object independently of native replica leases.
    virtual tl::expected<TaskHandle, ErrorCode> RemoveAsync(
        const ObjectKey& key, const TenantId& tenant_id) = 0;
    virtual tl::expected<TaskHandle, ErrorCode> BatchRemoveAsync(
        const std::vector<ObjectKey>& keys, const TenantId& tenant_id) = 0;

    // exists is non-null and stays valid until Wait returns.
    // Successful query items report 0 (missing) or 1 (present).
    virtual tl::expected<TaskHandle, ErrorCode> BatchExistsAsync(
        const std::vector<ObjectKey>& keys, const TenantId& tenant_id,
        std::vector<uint8_t>* exists) = 0;

    // targets[i] receives keys[i] in slice order. Sizes are writable
    // capacities. actual_sizes is non-null and stays valid until Wait returns;
    // successful items report the complete object size.
    virtual tl::expected<TaskHandle, ErrorCode> GetBatchAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const TenantId& tenant_id, std::vector<size_t>* actual_sizes) = 0;

    // Read exact byte ranges into the corresponding target slices.
    // actual_sizes has the same lifetime and reports bytes read per key.
    virtual tl::expected<TaskHandle, ErrorCode> GetRangesAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const std::vector<std::vector<size_t>>& source_offsets,
        const TenantId& tenant_id, std::vector<size_t>* actual_sizes) = 0;

    // Wait once per successful submission and consume the handle.
    // Return one status per input key, in input order. All target/MR access
    // must end before returning, including on timeout. Operation outputs are
    // valid for successful items after Wait returns.
    virtual std::vector<tl::expected<void, ErrorCode>> Wait(
        TaskHandle task) = 0;

    // Unsupported memory is skipped and returns success.
    // Keep the region alive until unregistration succeeds.
    virtual tl::expected<void, ErrorCode> RegisterMemory(
        void* addr, size_t length, const std::string& location) = 0;
    virtual tl::expected<void, ErrorCode> UnregisterMemory(void* addr) = 0;
};

}  // namespace mooncake
