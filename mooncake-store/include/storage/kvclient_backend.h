#pragma once

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_set>
#include <utility>

#include "kv_client.h"
#include "storage/external_store_backend.h"

namespace mooncake {

// ExternalStoreBackend implementation backed by the KVClient SDK.
class KvClientBackend final : public ExternalStoreBackend {
   public:
    ~KvClientBackend() override;
    tl::expected<void, ErrorCode> Init(
        const std::unordered_map<std::string, std::string>& options) override;
    tl::expected<void, ErrorCode> Shutdown() override;
    tl::expected<TaskHandle, ErrorCode> BatchExistsAsync(
        const std::vector<ObjectKey>& keys, const TenantId& tenant_id,
        std::vector<uint8_t>* exists) override;
    tl::expected<TaskHandle, ErrorCode> GetBatchAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const TenantId& tenant_id, std::vector<size_t>* actual_sizes) override;
    tl::expected<TaskHandle, ErrorCode> GetRangesAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const std::vector<std::vector<size_t>>& source_offsets,
        const TenantId& tenant_id, std::vector<size_t>* actual_sizes) override;

    std::vector<tl::expected<void, ErrorCode>> Wait(TaskHandle task) override;
    bool SupportsMemory(const device::PointerInfo& pointer_info) const override;
    tl::expected<void, ErrorCode> RegisterMemory(
        void* addr, size_t length, const std::string& location,
        const device::PointerInfo& pointer_info) override;
    tl::expected<void, ErrorCode> UnregisterMemory(void* addr) override;

   private:
    kv::CacheKey EncodeKey(const ObjectKey& key, const TenantId& tenant_id);
    kv::TaskResult WaitForCompletion(kv::TaskId task);
    tl::expected<TaskHandle, ErrorCode> SubmitLoads(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const std::vector<std::vector<size_t>>& source_offsets,
        const TenantId& tenant_id, std::vector<size_t>* actual_sizes);
    struct KvTaskContext {
        explicit KvTaskContext(
            std::function<std::vector<tl::expected<void, ErrorCode>>()> fn)
            : completion(std::move(fn)) {}

        std::function<std::vector<tl::expected<void, ErrorCode>>()> completion;
    };
    TaskHandle SaveTask(
        std::function<std::vector<tl::expected<void, ErrorCode>>()> completion);
    std::unique_ptr<kv::KvClient> sdk_;
    std::mutex tasks_mutex_;
    std::unordered_set<TaskHandle> tasks_;
    std::shared_mutex memory_mutex_;
    std::map<uintptr_t, kv::RegisteredMemory> regions_;
};

}  // namespace mooncake
