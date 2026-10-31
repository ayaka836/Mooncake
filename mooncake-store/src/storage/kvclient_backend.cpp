#include "storage/kvclient_backend.h"

#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <xxhash.h>
#include <utility>

#include "device/accelerator_registry.h"

namespace mooncake {
namespace {

ErrorCode ToError(const kv::Status& status) {
    switch (status.code) {
        case kv::StatusCode::OK:
            return ErrorCode::OK;
        case kv::StatusCode::ENTRY_KEY_NOT_FOUND:
        case kv::StatusCode::ENTRY_KEY_NOT_EXIST:
        case kv::StatusCode::ENTRY_DATA_NOT_EXIST:
        case kv::StatusCode::CQE_KEY_NOT_EXIST:
            return ErrorCode::OBJECT_NOT_FOUND;
        case kv::StatusCode::INVALID_ARGUMENT:
        case kv::StatusCode::BUFFER_NOT_REGISTERED:
        case kv::StatusCode::TASK_NOT_FOUND:
        case kv::StatusCode::CQE_INVALID_FIELD_IN_COMMAND:
            return ErrorCode::INVALID_PARAMS;
        case kv::StatusCode::UNSUPPORTED:
        case kv::StatusCode::BUFFER_NOT_SUPPORTED:
        case kv::StatusCode::CQE_INVALID_COMMAND_OPCODE:
            return ErrorCode::NOT_SUPPORTED;
        case kv::StatusCode::NOT_INITIALIZED:
        case kv::StatusCode::RESOURCE_BUSY:
        case kv::StatusCode::CQE_RESOURCE_BUSY:
            return ErrorCode::UNAVAILABLE_IN_CURRENT_STATUS;
        case kv::StatusCode::INTERNAL_ERROR:
        case kv::StatusCode::CQE_INTERNAL_ERROR:
            return ErrorCode::INTERNAL_ERROR;
        default:
            // Generic NOT_FOUND is not proof that the object is missing.
            return ErrorCode::TRANSFER_FAIL;
    }
}

}  // namespace

kv::CacheKey KvClientBackend::EncodeKey(const ObjectKey& key,
                                        const TenantId& tenant_id) {
    // Scope by tenant before encoding the SDK key.
    const auto scoped_key = tenant_id.MakeScopedKey(key);
    const uint64_t hash = XXH64(scoped_key.data(), scoped_key.size(), 0);
    XXH64_canonical_t canonical;
    XXH64_canonicalFromHash(&canonical, hash);
    kv::CacheKey encoded;
    static_assert(kv::kCacheKeySizeBytes == sizeof(canonical.digest));
    std::memcpy(encoded.data(), canonical.digest, encoded.size());
    return encoded;
}

KvClientBackend::~KvClientBackend() { (void)Shutdown(); }

tl::expected<void, ErrorCode> KvClientBackend::Shutdown() {
    // Collect outstanding tasks before shutting down the SDK.
    while (!tasks_.empty()) {
        (void)Wait(*tasks_.begin());
    }
    if (!sdk_) return {};
    // SDK shutdown also releases registered memory.
    auto status = sdk_->Shutdown();
    sdk_.reset();
    regions_.clear();
    if (!status.ok()) return tl::unexpected(ToError(status));
    return {};
}

ExternalStoreBackend::TaskHandle KvClientBackend::SaveTask(
    std::function<std::vector<tl::expected<void, ErrorCode>>()> completion) {
    auto* task = new KvTaskContext(std::move(completion));
    std::lock_guard lock(tasks_mutex_);
    tasks_.insert(task);
    return task;
}

std::vector<tl::expected<void, ErrorCode>> KvClientBackend::Wait(
    TaskHandle task) {
    std::unique_ptr<KvTaskContext> context(static_cast<KvTaskContext*>(task));
    {
        std::lock_guard lock(tasks_mutex_);
        tasks_.erase(task);
    }
    return context->completion();
}

tl::expected<void, ErrorCode> KvClientBackend::Init(
    const std::unordered_map<std::string, std::string>& options) {
    const auto config = options.find("config_path");
    if (config == options.end()) {
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    sdk_ = kv::CreateKvClient();
    auto status = sdk_->Init(config->second);
    if (!status.ok()) return tl::unexpected(ToError(status));
    return {};
}

bool KvClientBackend::SupportsMemory(
    const device::PointerInfo& pointer_info) const {
    // ASU supports device memory for business data.
    return pointer_info.kind == device::MemoryKind::kDevice;
}

tl::expected<void, ErrorCode> KvClientBackend::RegisterMemory(
    void* addr, size_t length, [[maybe_unused]] const std::string& location,
    const device::PointerInfo& pointer_info) {
    std::unique_lock lock(memory_mutex_);
    if (!SupportsMemory(pointer_info)) {
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    }
    kv::MemoryRegion region;
    region.memoryType = kv::MemoryType::DEVICE;
    region.addr = reinterpret_cast<uintptr_t>(addr);
    region.size = length;
    region.deviceId = pointer_info.device_id;
    if (regions_.find(region.addr) != regions_.end()) {
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    std::vector<kv::RegisteredMemory> registered;
    auto status = sdk_->RegisterRegions({region}, registered);
    if (!status.ok()) return tl::unexpected(ToError(status));
    regions_.emplace(region.addr, registered.front());
    return {};
}

tl::expected<void, ErrorCode> KvClientBackend::UnregisterMemory(void* addr) {
    std::unique_lock lock(memory_mutex_);
    auto iter = regions_.find(reinterpret_cast<uintptr_t>(addr));
    // Host buffers may be registered only with TE.
    if (iter == regions_.end()) {
        return {};
    }
    auto status = sdk_->UnregisterRegions({iter->second.handle});
    if (!status.ok()) return tl::unexpected(ToError(status));
    regions_.erase(iter);
    return {};
}

kv::TaskResult KvClientBackend::WaitForCompletion(kv::TaskId task) {
    kv::TaskResult result;
    // The SDK must drain timed-out work before returning.
    auto status = sdk_->Wait(task, 0, result);  // Use the SDK default timeout.
    result.status = status;
    if (!status.ok() && status.code != kv::StatusCode::PARTIAL_FAILED) {
        // Ignore provisional entry results on a task-level failure.
        result.entryStatus.clear();
        result.queryResult.reset();
    }
    return result;
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchExistsAsync(const std::vector<ObjectKey>& keys,
                                  const TenantId& tenant_id,
                                  std::vector<uint8_t>* exists) {
    if (keys.empty()) {
        return SaveTask([exists] {
            exists->clear();
            return std::vector<tl::expected<void, ErrorCode>>{};
        });
    }
    std::vector<kv::CacheKey> sdk_keys;
    for (const auto& key : keys) sdk_keys.push_back(EncodeKey(key, tenant_id));
    kv::TaskId task;
    auto submitted = sdk_->QueryAsync(sdk_keys, task);
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task, exists, count = keys.size()] {
        auto completed = WaitForCompletion(task);
        std::vector<tl::expected<void, ErrorCode>> results(count);
        exists->assign(count, 0);
        for (size_t j = 0; j < count; ++j) {
            const auto& status = completed.entryStatus.empty()
                                     ? completed.status
                                     : completed.entryStatus[j];
            if (status.code == kv::StatusCode::ENTRY_KEY_EXIST) {
                (*exists)[j] = 1;
            } else if (ToError(status) == ErrorCode::OBJECT_NOT_FOUND &&
                       !status.ok()) {
                (*exists)[j] = 0;
            } else if (!status.ok()) {
                results[j] = tl::unexpected(ToError(status));
            } else {
                (*exists)[j] = completed.queryResult->exists[j] != 0;
            }
        }
        return results;
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::GetBatchAsync(const std::vector<ObjectKey>& keys,
                               const std::vector<std::vector<Slice>>& targets,
                               const TenantId& tenant_id,
                               std::vector<size_t>* actual_sizes) {
    std::vector<std::vector<size_t>> offsets(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        size_t offset = 0;
        for (const auto& slice : targets[i]) {
            offsets[i].push_back(offset);
            offset += slice.size;
        }
    }
    // MC supplies capacities; this adapter currently requires exact lengths.
    return SubmitLoads(keys, targets, offsets, tenant_id, actual_sizes);
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::GetRangesAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets,
    const std::vector<std::vector<size_t>>& source_offsets,
    const TenantId& tenant_id, std::vector<size_t>* actual_sizes) {
    return SubmitLoads(keys, targets, source_offsets, tenant_id, actual_sizes);
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::SubmitLoads(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets,
    const std::vector<std::vector<size_t>>& source_offsets,
    const TenantId& tenant_id, std::vector<size_t>* actual_sizes) {
    if (keys.empty()) {
        return SaveTask([actual_sizes] {
            actual_sizes->clear();
            return std::vector<tl::expected<void, ErrorCode>>{};
        });
    }
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    std::vector<kv::KVBuffer> entries;
    std::vector<size_t> indices;
    std::vector<size_t> lengths;
    auto runtime = device::GetAcceleratorRegistry().RuntimeAccelerators();
    // Keep registrations stable until the SDK accepts the descriptors.
    std::shared_lock lock(memory_mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto key = EncodeKey(keys[i], tenant_id);
        std::vector<kv::KVBuffer> object_entries;
        if (targets[i].empty()) {
            results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
            continue;
        }
        for (size_t j = 0; j < targets[i].size(); ++j) {
            const auto& slice = targets[i][j];
            const auto object_offset = source_offsets[i][j];
            device::PointerInfo info{.kind = device::MemoryKind::kHost,
                                     .device_id = -1};
            runtime.FindDeviceForPointer(slice.ptr, &info);
            if (!SupportsMemory(info)) {
                results[i] = tl::unexpected(ErrorCode::NOT_SUPPORTED);
                break;
            }
            const auto addr = reinterpret_cast<uintptr_t>(slice.ptr);
            auto region = regions_.upper_bound(addr);
            if (region == regions_.begin()) {
                results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
                break;
            }
            const auto& registered = std::prev(region)->second;
            const auto region_offset = addr - registered.region.addr;
            if (region_offset > registered.region.size ||
                slice.size > registered.region.size - region_offset) {
                results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
                break;
            }
            // SDK offsets and lengths are limited to uint32.
            if (object_offset > std::numeric_limits<uint32_t>::max() ||
                slice.size > std::numeric_limits<uint32_t>::max()) {
                results[i] = tl::unexpected(ErrorCode::NOT_SUPPORTED);
                break;
            }
            kv::KVBuffer entry{};
            entry.key = key;
            entry.buffer.region = registered.region;
            entry.buffer.handle = registered.handle;
            entry.buffer.region.addr = addr;
            entry.buffer.region.size = slice.size;
            // Offset is relative to the object, not the registered region.
            entry.offset = static_cast<uint32_t>(object_offset);
            object_entries.push_back(entry);
        }
        // Submit an input key only when all its slices are valid.
        if (!results[i]) continue;
        for (const auto& entry : object_entries) {
            entries.push_back(entry);
            indices.push_back(i);
            lengths.push_back(entry.buffer.region.size);
        }
    }
    if (entries.empty()) {
        lock.unlock();
        return SaveTask([actual_sizes, results = std::move(results)]() mutable {
            actual_sizes->assign(results.size(), 0);
            return std::move(results);
        });
    }
    kv::TaskId sdk_task;
    auto submitted = sdk_->BatchLoadAsync(entries, sdk_task);
    lock.unlock();
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, sdk_task, actual_sizes, indices = std::move(indices),
                     lengths = std::move(lengths),
                     results = std::move(results)]() mutable {
        auto completed = WaitForCompletion(sdk_task);
        actual_sizes->assign(results.size(), 0);
        for (size_t j = 0; j < indices.size(); ++j) {
            const auto& status = completed.entryStatus.empty()
                                     ? completed.status
                                     : completed.entryStatus[j];
            auto& result = results[indices[j]];
            if (!status.ok()) {
                if (result) result = tl::unexpected(ToError(status));
            } else if (result) {
                (*actual_sizes)[indices[j]] += lengths[j];
            }
        }
        return std::move(results);
    });
}

}  // namespace mooncake
