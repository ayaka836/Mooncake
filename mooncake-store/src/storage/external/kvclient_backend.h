#pragma once

#include <functional>
#include <map>
#include <memory>
#include <shared_mutex>
#include <utility>

#include "kv_client.h"
#include "storage/external/external_store_backend.h"

namespace mooncake {

// ExternalStoreBackend implementation backed by the KVClient SDK.
class KvClientBackend final : public ExternalStoreBackend {
   public:
    ~KvClientBackend() override;
    tl::expected<void, ErrorCode> Init(
        const std::unordered_map<std::string, std::string>& options,
        const TenantId& tenant_id) override;
    tl::expected<void, ErrorCode> Shutdown() override;
    tl::expected<TaskHandle, ErrorCode> BatchPutAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& sources) override;
    tl::expected<TaskHandle, ErrorCode> BatchPutRangesAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& sources,
        const std::vector<std::vector<size_t>>& target_offsets) override;
    tl::expected<TaskHandle, ErrorCode> BatchRemoveAsync(
        const std::vector<ObjectKey>& keys) override;

    tl::expected<TaskHandle, ErrorCode> BatchExistsAsync(
        const std::vector<ObjectKey>& keys) override;
    tl::expected<TaskHandle, ErrorCode> BatchGetAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets) override;
    tl::expected<TaskHandle, ErrorCode> BatchGetRangesAsync(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const std::vector<std::vector<size_t>>& source_offsets) override;

    std::vector<tl::expected<void, ErrorCode>> Wait(
        TaskHandle task, TaskOutput& output) override;
    tl::expected<void, ErrorCode> RegisterMemory(
        void* addr, size_t length, const std::string& location) override;
    tl::expected<void, ErrorCode> UnregisterMemory(void* addr) override;

   private:
    tl::expected<const kv::RegisteredMemory*, ErrorCode> FindRegisteredMemory(
        const Slice& slice) const;
    // Append one descriptor per slice in input order.
    tl::expected<void, ErrorCode> AppendStoreEntries(
        const kv::CacheKey& key, const std::vector<Slice>& sources,
        std::vector<kv::KVBuffer>& entries,
        const std::vector<size_t>* target_offsets = nullptr);

    // Build descriptors and map each entry back to its input key index.
    std::vector<kv::KVBuffer> BuildStoreEntries(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& sources,
        std::vector<size_t>& indices,
        std::vector<tl::expected<void, ErrorCode>>& results,
        const std::vector<std::vector<size_t>>* target_offsets = nullptr);

    // Reduce SDK entry results to one completion status per input key.
    std::vector<tl::expected<void, ErrorCode>> WaitForKeyResults(
        kv::TaskId task, const std::vector<size_t>& indices,
        std::vector<tl::expected<void, ErrorCode>> results);

    kv::TaskResult WaitForCompletion(kv::TaskId task);
    tl::expected<TaskHandle, ErrorCode> SubmitLoads(
        const std::vector<ObjectKey>& keys,
        const std::vector<std::vector<Slice>>& targets,
        const std::vector<std::vector<size_t>>& source_offsets);
    using Completion =
        std::function<std::vector<tl::expected<void, ErrorCode>>(TaskOutput&)>;
    std::unique_ptr<kv::KvClient> kv_client_;
    kv::IoOptions io_options_;
    std::shared_mutex memory_mutex_;
    std::map<uintptr_t, kv::RegisteredMemory> regions_;
};

}  // namespace mooncake
