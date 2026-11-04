#include "storage/external/kvclient_backend.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <glog/logging.h>
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

tl::expected<kv::CacheKey, ErrorCode> KvClientBackend::EncodeKey(
    const ObjectKey& key, const TenantId& tenant_id) {
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
    // SDK shutdown drains tasks and releases registered memory.
    auto status = kv_client_ ? kv_client_->Shutdown() : kv::Status::OK();
    kv_client_.reset();
    for (auto task : tasks_) {
        delete static_cast<KvTaskContext*>(task);
    }
    tasks_.clear();
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
        LOG(ERROR) << "KVClient backend requires config_path";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    kv_client_ = kv::CreateKvClient();
    auto status = kv_client_->Init(config->second);
    if (!status.ok()) {
        LOG(ERROR) << "Failed to initialize KVClient backend, config="
                   << config->second
                   << ", status=" << static_cast<int>(status.code)
                   << ", error=" << status.message;
        return tl::unexpected(ToError(status));
    }
    return {};
}

tl::expected<void, ErrorCode> KvClientBackend::RegisterMemory(
    void* addr, size_t length, [[maybe_unused]] const std::string& location) {
    device::PointerInfo pointer_info{.kind = device::MemoryKind::kHost,
                                     .device_id = -1};
    auto runtime = device::GetAcceleratorRegistry().RuntimeAccelerators();
    runtime.FindDeviceForPointer(addr, &pointer_info);
    // ASU supports device memory for business data.
    if (pointer_info.kind != device::MemoryKind::kDevice) return {};
    std::unique_lock lock(memory_mutex_);
    kv::MemoryRegion region;
    region.memoryType = kv::MemoryType::DEVICE;
    region.addr = reinterpret_cast<uintptr_t>(addr);
    region.size = length;
    region.deviceId = pointer_info.device_id;
    if (regions_.find(region.addr) != regions_.end()) {
        LOG(WARNING) << "KVClient memory is already registered, addr=" << addr
                     << ", size=" << length << ", device=" << region.deviceId;
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    std::vector<kv::RegisteredMemory> registered;
    auto status = kv_client_->RegisterRegions({region}, registered);
    if (!status.ok()) {
        LOG(ERROR) << "Failed to register KVClient memory, addr=" << addr
                   << ", size=" << length << ", device=" << region.deviceId
                   << ", status=" << static_cast<int>(status.code)
                   << ", error=" << status.message;
        return tl::unexpected(ToError(status));
    }
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
    auto status = kv_client_->UnregisterRegions({iter->second.handle});
    if (!status.ok()) {
        LOG(ERROR) << "Failed to unregister KVClient memory, addr=" << addr
                   << ", device=" << iter->second.region.deviceId
                   << ", status=" << static_cast<int>(status.code)
                   << ", error=" << status.message;
        return tl::unexpected(ToError(status));
    }
    regions_.erase(iter);
    return {};
}

kv::TaskResult KvClientBackend::WaitForCompletion(kv::TaskId task) {
    kv::TaskResult result;
    // The SDK must drain timed-out work before returning.
    auto status =
        kv_client_->Wait(task, 0, result);  // Use the SDK default timeout.
    result.status = status;
    if (!status.ok() && status.code != kv::StatusCode::PARTIAL_FAILED &&
        ToError(status) != ErrorCode::OBJECT_NOT_FOUND) {
        LOG(WARNING) << "KVClient task failed, task=" << task
                     << ", status=" << static_cast<int>(status.code)
                     << ", error=" << status.message;
    } else {
        for (size_t i = 0; i < result.entryStatus.size(); ++i) {
            const auto& entry = result.entryStatus[i];
            if (entry.ok() || entry.code == kv::StatusCode::ENTRY_KEY_EXIST ||
                ToError(entry) == ErrorCode::OBJECT_NOT_FOUND)
                continue;
            LOG(WARNING) << "KVClient entry failed, task=" << task
                         << ", entry=" << i
                         << ", status=" << static_cast<int>(entry.code)
                         << ", error=" << entry.message;
        }
    }
    return result;
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchExistsAsync(const std::vector<ObjectKey>& keys,
                                  const TenantId& tenant_id,
                                  std::vector<uint8_t>* exists) {
    std::vector<kv::CacheKey> encoded_keys;
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    encoded_keys.reserve(keys.size());
    indices.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        auto key = EncodeKey(keys[i], tenant_id);
        if (!key) {
            LOG(WARNING) << "Invalid KVClient query key, index=" << i
                         << ", key=" << keys[i]
                         << ", error=" << static_cast<int>(key.error());
            results[i] = tl::unexpected(key.error());
            continue;
        }
        encoded_keys.push_back(*key);
        indices.push_back(i);
    }
    kv::TaskId task = kv::kInvalidTaskId;
    if (!encoded_keys.empty()) {
        auto submitted = kv_client_->QueryAsync(encoded_keys, task);
        if (!submitted.ok()) {
            LOG(WARNING) << "Failed to submit KVClient query, keys="
                         << encoded_keys.size()
                         << ", status=" << static_cast<int>(submitted.code)
                         << ", error=" << submitted.message;
            return tl::unexpected(ToError(submitted));
        }
    }
    return SaveTask([this, task, exists, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        kv::TaskResult completed;
        if (task != kv::kInvalidTaskId) completed = WaitForCompletion(task);
        exists->assign(results.size(), 0);
        for (size_t j = 0; j < indices.size(); ++j) {
            const auto index = indices[j];
            const auto& status = completed.entryStatus.empty()
                                     ? completed.status
                                     : completed.entryStatus[j];
            if (status.code == kv::StatusCode::ENTRY_KEY_EXIST) {
                (*exists)[index] = 1;
            } else if (ToError(status) == ErrorCode::OBJECT_NOT_FOUND) {
                (*exists)[index] = 0;
            } else if (!status.ok()) {
                results[index] = tl::unexpected(ToError(status));
            }
        }
        return std::move(results);
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchGetAsync(const std::vector<ObjectKey>& keys,
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
KvClientBackend::BatchGetRangesAsync(
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
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    std::vector<kv::KVBuffer> entries;
    std::vector<size_t> indices;
    std::vector<size_t> lengths;
    // Keep registrations stable until the SDK accepts the descriptors.
    std::shared_lock lock(memory_mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        auto key = EncodeKey(keys[i], tenant_id);
        if (!key) {
            LOG(WARNING) << "Invalid KVClient load key, index=" << i
                         << ", key=" << keys[i]
                         << ", error=" << static_cast<int>(key.error());
            results[i] = tl::unexpected(key.error());
            continue;
        }
        std::vector<kv::KVBuffer> object_entries;
        if (targets[i].empty()) {
            LOG(WARNING) << "Missing KVClient load targets for key=" << keys[i];
            results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
            continue;
        }
        for (size_t j = 0; j < targets[i].size(); ++j) {
            const auto& slice = targets[i][j];
            // Range offsets are relative to the object.
            const auto source_offset = source_offsets[i][j];
            if (source_offset > std::numeric_limits<uint32_t>::max()) {
                LOG(WARNING)
                    << "KVClient load exceeds SDK offset limits for key="
                    << keys[i] << ", slice=" << j
                    << ", offset=" << source_offset
                    << ", source_offset=" << source_offset;
                results[i] = tl::unexpected(ErrorCode::NOT_SUPPORTED);
                break;
            }
            const auto object_offset = source_offset;
            const auto addr = reinterpret_cast<uintptr_t>(slice.ptr);
            auto registered = FindRegisteredMemory(slice);
            if (!registered) {
                LOG(WARNING)
                    << "Invalid KVClient load target for key=" << keys[i]
                    << ", slice=" << j << ", addr=" << slice.ptr
                    << ", size=" << slice.size
                    << ", error=" << static_cast<int>(registered.error());
                results[i] = tl::unexpected(registered.error());
                break;
            }
            // SDK offsets and lengths are limited to uint32.
            if (slice.size > std::numeric_limits<uint32_t>::max()) {
                LOG(WARNING)
                    << "KVClient load exceeds SDK length limits for key="
                    << keys[i] << ", slice=" << j << ", size=" << slice.size;
                results[i] = tl::unexpected(ErrorCode::NOT_SUPPORTED);
                break;
            }
            kv::KVBuffer entry{};
            entry.key = *key;
            entry.buffer.region = (*registered)->region;
            entry.buffer.handle = (*registered)->handle;
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
    kv::TaskId sdk_task = kv::kInvalidTaskId;
    auto submitted = kv::Status::OK();
    if (!entries.empty())
        submitted = kv_client_->BatchLoadAsync(entries, sdk_task);
    lock.unlock();
    if (!submitted.ok()) {
        LOG(WARNING) << "Failed to submit KVClient load, entries="
                     << entries.size()
                     << ", status=" << static_cast<int>(submitted.code)
                     << ", error=" << submitted.message;
        return tl::unexpected(ToError(submitted));
    }
    return SaveTask([this, sdk_task, actual_sizes, indices = std::move(indices),
                     lengths = std::move(lengths),
                     results = std::move(results)]() mutable {
        kv::TaskResult completed;
        if (sdk_task != kv::kInvalidTaskId)
            completed = WaitForCompletion(sdk_task);
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

tl::expected<const kv::RegisteredMemory*, ErrorCode>
KvClientBackend::FindRegisteredMemory(const Slice& slice) const {
    const auto addr = reinterpret_cast<uintptr_t>(slice.ptr);
    auto region = regions_.upper_bound(addr);
    if (region == regions_.begin())
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    const auto& registered = std::prev(region)->second;
    const auto offset = addr - registered.region.addr;
    if (offset > registered.region.size || slice.size == 0 ||
        slice.size > registered.region.size - offset)
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    if (registered.region.memoryType != kv::MemoryType::DEVICE)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    return &registered;
}

}  // namespace mooncake
