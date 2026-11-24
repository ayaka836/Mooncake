#include "storage/external/kvclient_backend.h"

#include <algorithm>
#include <glog/logging.h>
#include <iterator>
#include <limits>
#include <mutex>
#include <utility>

#include "device/accelerator_registry.h"

namespace mooncake {

namespace {

ErrorCode ToError(const kv::Status& status) {
    switch (status.code) {
        case kv::StatusCode::OK:
        case kv::StatusCode::ENTRY_KEY_EXIST:
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

KvClientBackend::~KvClientBackend() { (void)Shutdown(); }

tl::expected<void, ErrorCode> KvClientBackend::Shutdown() {
    // SDK shutdown drains tasks and releases registered memory.
    auto status = kv_client_ ? kv_client_->Shutdown() : kv::Status::OK();
    kv_client_.reset();
    regions_.clear();
    if (!status.ok()) return tl::unexpected(ToError(status));
    return {};
}

std::vector<tl::expected<void, ErrorCode>> KvClientBackend::Wait(
    TaskHandle task, TaskOutput& output) {
    std::unique_ptr<Completion> completion(static_cast<Completion*>(task));
    output = {};
    return (*completion)(output);
}

tl::expected<void, ErrorCode> KvClientBackend::Init(
    const std::unordered_map<std::string, std::string>& options,
    const TenantId& tenant_id) {
    const auto config = options.find("config_path");
    if (config == options.end()) {
        LOG(ERROR) << "KVClient backend requires config_path";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    kv::KvClientConfig client_config;
    auto status =
        kv::KvClientConfig::LoadFromFile(config->second, client_config);
    if (!status.ok()) {
        LOG(ERROR) << "Failed to load KVClient config, path=" << config->second
                   << ", status=" << static_cast<int>(status.code)
                   << ", error=" << status.message;
        return tl::unexpected(ToError(status));
    }
    kv_client_ = kv::CreateKvClient();
    status = kv_client_->Init(client_config);
    if (!status.ok()) {
        LOG(ERROR) << "Failed to initialize KVClient backend, config="
                   << config->second
                   << ", status=" << static_cast<int>(status.code)
                   << ", error=" << status.message;
        return tl::unexpected(ToError(status));
    }
    io_options_.kvns = tenant_id.value();
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
    if (!status.ok() && status.code != kv::StatusCode::PARTIAL_FAILED) {
        const auto error = ToError(status);
        if (error != ErrorCode::OK && error != ErrorCode::OBJECT_NOT_FOUND) {
            LOG(WARNING) << "KVClient task failed, task=" << task
                         << ", status=" << static_cast<int>(status.code)
                         << ", error=" << status.message;
        }
    } else {
        for (size_t i = 0; i < result.entryStatus.size(); ++i) {
            const auto& entry = result.entryStatus[i];
            const auto error = ToError(entry);
            if (error == ErrorCode::OK || error == ErrorCode::OBJECT_NOT_FOUND)
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
KvClientBackend::BatchExistsAsync(const std::vector<ObjectKey>& keys) {
    std::vector<kv::CacheKey> query_keys;
    query_keys.reserve(keys.size());
    for (const auto& key : keys) {
        query_keys.push_back(kv::CacheKey::FromString(key));
    }
    kv::TaskId task = kv::kInvalidTaskId;
    if (!query_keys.empty()) {
        auto submitted = kv_client_->QueryAsync(query_keys, task, io_options_);
        if (!submitted.ok()) {
            LOG(WARNING) << "Failed to submit KVClient query, keys="
                         << query_keys.size()
                         << ", status=" << static_cast<int>(submitted.code)
                         << ", error=" << submitted.message;
            return tl::unexpected(ToError(submitted));
        }
    }
    return new Completion(
        [this, task, count = keys.size()](TaskOutput& output) {
            kv::TaskResult completed;
            if (task != kv::kInvalidTaskId) completed = WaitForCompletion(task);
            std::vector<tl::expected<void, ErrorCode>> results(count);
            output.exists.assign(count, 0);
            // Only OK and PARTIAL_FAILED guarantee valid per-entry results.
            const bool has_entries =
                completed.status.ok() ||
                completed.status.code == kv::StatusCode::PARTIAL_FAILED;
            for (size_t i = 0; i < count; ++i) {
                const auto& status =
                    has_entries ? completed.entryStatus[i] : completed.status;
                const auto error = ToError(status);
                if (error == ErrorCode::OK) {
                    output.exists[i] =
                        status.code == kv::StatusCode::ENTRY_KEY_EXIST;
                } else if (error != ErrorCode::OBJECT_NOT_FOUND) {
                    results[i] = tl::unexpected(error);
                }
            }
            return results;
        });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchGetAsync(const std::vector<ObjectKey>& keys,
                               const std::vector<std::vector<Slice>>& targets) {
    std::vector<std::vector<size_t>> offsets(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        size_t offset = 0;
        for (const auto& slice : targets[i]) {
            offsets[i].push_back(offset);
            offset += slice.size;
        }
    }
    // MC supplies capacities; this adapter currently requires exact lengths.
    return SubmitLoads(keys, targets, offsets);
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchGetRangesAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets,
    const std::vector<std::vector<size_t>>& source_offsets) {
    return SubmitLoads(keys, targets, source_offsets);
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::SubmitLoads(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets,
    const std::vector<std::vector<size_t>>& source_offsets) {
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    std::vector<kv::KVBuffer> entries;
    std::vector<size_t> indices;
    std::vector<size_t> lengths;
    // Keep registrations stable until the SDK accepts the descriptors.
    std::shared_lock lock(memory_mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        std::vector<kv::KVBuffer> object_entries;
        if (targets[i].empty()) {
            LOG(WARNING) << "Missing KVClient load targets for key=" << keys[i];
            results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
            continue;
        }
        for (size_t j = 0; j < targets[i].size(); ++j) {
            const auto& slice = targets[i][j];
            if (slice.size == 0) continue;
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
            entry.key = kv::CacheKey::FromString(keys[i]);
            entry.buffer.region = (*registered)->region;
            entry.buffer.handle = (*registered)->handle;
            entry.buffer.region.addr = addr;
            entry.buffer.region.size = slice.size;
            // Offset is relative to the object, not the registered region.
            entry.offset = static_cast<uint32_t>(object_offset);
            object_entries.push_back(std::move(entry));
        }
        // Submit an input key only when all its slices are valid.
        if (!results[i]) continue;
        for (auto& entry : object_entries) {
            indices.push_back(i);
            lengths.push_back(entry.buffer.region.size);
            entries.push_back(std::move(entry));
        }
    }
    kv::TaskId sdk_task = kv::kInvalidTaskId;
    auto submitted = kv::Status::OK();
    if (!entries.empty()) {
        submitted = kv_client_->LoadAsync(entries, sdk_task, io_options_);
    }
    lock.unlock();
    if (!submitted.ok()) {
        LOG(WARNING) << "Failed to submit KVClient load, entries="
                     << entries.size()
                     << ", status=" << static_cast<int>(submitted.code)
                     << ", error=" << submitted.message;
        return tl::unexpected(ToError(submitted));
    }
    return new Completion([this, sdk_task, indices = std::move(indices),
                           lengths = std::move(lengths),
                           results =
                               std::move(results)](TaskOutput& output) mutable {
        kv::TaskResult completed;
        if (sdk_task != kv::kInvalidTaskId)
            completed = WaitForCompletion(sdk_task);
        output.actual_sizes.assign(results.size(), 0);
        // Only OK and PARTIAL_FAILED guarantee valid per-entry results.
        const bool has_entries =
            completed.status.ok() ||
            completed.status.code == kv::StatusCode::PARTIAL_FAILED;
        for (size_t j = 0; j < indices.size(); ++j) {
            const auto& status =
                has_entries ? completed.entryStatus[j] : completed.status;
            const auto error = ToError(status);
            auto& result = results[indices[j]];
            if (error != ErrorCode::OK) {
                if (result) result = tl::unexpected(error);
            } else if (result) {
                output.actual_sizes[indices[j]] += lengths[j];
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

tl::expected<void, ErrorCode> KvClientBackend::AppendStoreEntries(
    const kv::CacheKey& key, const std::vector<Slice>& slices,
    std::vector<kv::KVBuffer>& entries,
    const std::vector<size_t>* destination_offsets) {
    if (slices.empty()) {
        LOG(WARNING) << "Missing KVClient store sources";
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    if (destination_offsets && destination_offsets->size() != slices.size()) {
        LOG(WARNING) << "KVClient store range count mismatch, slices="
                     << slices.size()
                     << ", offsets=" << destination_offsets->size();
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    // Preserve slice order even when the source addresses are noncontiguous.
    uint64_t logical_offset = 0;
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto& slice = slices[i];
        if (destination_offsets) logical_offset = (*destination_offsets)[i];
        // Check before converting offsets and lengths to SDK descriptors.
        if (logical_offset > std::numeric_limits<uint32_t>::max() ||
            slice.size > std::numeric_limits<uint32_t>::max()) {
            LOG(WARNING) << "KVClient store exceeds SDK offset or length limits"
                         << ", slice=" << i << ", offset=" << logical_offset
                         << ", size=" << slice.size;
            return tl::unexpected(ErrorCode::NOT_SUPPORTED);
        }
        kv::RegisteredMemory registered;
        {
            std::shared_lock lock(memory_mutex_);
            auto region = FindRegisteredMemory(slice);
            if (!region) {
                LOG(WARNING)
                    << "Invalid KVClient store source, slice=" << i
                    << ", addr=" << slice.ptr << ", size=" << slice.size
                    << ", error=" << static_cast<int>(region.error());
                return tl::unexpected(region.error());
            }
            // Keep a snapshot; the map entry may be erased after unlocking.
            registered = **region;
        }
        kv::KVBuffer entry{};
        entry.key = key;
        entry.buffer.region = registered.region;
        entry.buffer.region.addr = reinterpret_cast<uintptr_t>(slice.ptr);
        entry.buffer.region.size = slice.size;
        entry.buffer.handle = registered.handle;
        entry.offset = static_cast<uint32_t>(logical_offset);
        entries.push_back(entry);
        logical_offset += slice.size;
    }
    return {};
}

std::vector<kv::KVBuffer> KvClientBackend::BuildStoreEntries(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& batched_slices,
    std::vector<size_t>& indices,
    std::vector<tl::expected<void, ErrorCode>>& results,
    const std::vector<std::vector<size_t>>* destination_offsets) {
    std::vector<kv::KVBuffer> entries;
    results.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto begin = entries.size();
        results[i] = AppendStoreEntries(
            kv::CacheKey::FromString(keys[i]), batched_slices[i], entries,
            destination_offsets ? &(*destination_offsets)[i] : nullptr);
        if (!results[i]) {
            LOG(WARNING) << "Invalid KVClient store sources, index=" << i
                         << ", key=" << keys[i]
                         << ", error=" << static_cast<int>(results[i].error());
            // Do not submit a partial set of descriptors for this input key.
            entries.resize(begin);
            continue;
        }
        indices.insert(indices.end(), entries.size() - begin, i);
    }
    return entries;
}

std::vector<tl::expected<void, ErrorCode>> KvClientBackend::WaitForKeyResults(
    kv::TaskId task, const std::vector<size_t>& indices,
    std::vector<tl::expected<void, ErrorCode>> results) {
    const auto completed = WaitForCompletion(task);
    const auto& status = completed.status;
    const bool has_entries =
        status.ok() || status.code == kv::StatusCode::PARTIAL_FAILED;
    if (has_entries && completed.entryStatus.size() != indices.size()) {
        LOG(ERROR) << "KVClient result count mismatch: expected="
                   << indices.size()
                   << ", actual=" << completed.entryStatus.size();
        for (const auto index : indices) {
            if (results[index])
                results[index] = tl::unexpected(ErrorCode::RPC_FAIL);
        }
        return results;
    }
    // Multiple SDK entries can belong to one input key; any failure wins.
    for (size_t j = 0; j < indices.size(); ++j) {
        const auto error =
            ToError(has_entries ? completed.entryStatus[j] : status);
        auto& item = results[indices[j]];
        if (item && error != ErrorCode::OK) item = tl::unexpected(error);
    }
    return results;
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchPutAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& batched_slices) {
    if (keys.size() != batched_slices.size()) {
        LOG(WARNING) << "Invalid KVClient batch store arguments, keys="
                     << keys.size()
                     << ", batched_slices=" << batched_slices.size();
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results;
    auto entries = BuildStoreEntries(keys, batched_slices, indices, results);
    if (entries.empty()) {
        return new Completion(
            [results = std::move(results)](TaskOutput&) mutable {
                return std::move(results);
            });
    }
    kv::TaskId task = kv::kInvalidTaskId;
    auto submitted = kv_client_->StoreAsync(entries, task, io_options_);
    if (!submitted.ok()) {
        LOG(WARNING) << "Failed to submit KVClient store, entries="
                     << entries.size()
                     << ", status=" << static_cast<int>(submitted.code)
                     << ", error=" << submitted.message;
        return tl::unexpected(ToError(submitted));
    }
    return new Completion([this, task, indices = std::move(indices),
                           results = std::move(results)](TaskOutput&) mutable {
        return WaitForKeyResults(task, indices, std::move(results));
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchPutRangesAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& sources,
    const std::vector<std::vector<size_t>>& destination_offsets) {
    if (keys.size() != sources.size() ||
        keys.size() != destination_offsets.size()) {
        LOG(WARNING) << "Invalid KVClient batch range store arguments, keys="
                     << keys.size() << ", sources=" << sources.size()
                     << ", offsets=" << destination_offsets.size();
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results;
    auto entries = BuildStoreEntries(keys, sources, indices, results,
                                     &destination_offsets);
    if (entries.empty()) {
        return new Completion(
            [results = std::move(results)](TaskOutput&) mutable {
                return std::move(results);
            });
    }
    kv::TaskId task = kv::kInvalidTaskId;
    auto submitted = kv_client_->StoreAsync(entries, task, io_options_);
    if (!submitted.ok()) {
        LOG(WARNING) << "Failed to submit KVClient store, entries="
                     << entries.size()
                     << ", status=" << static_cast<int>(submitted.code)
                     << ", error=" << submitted.message;
        return tl::unexpected(ToError(submitted));
    }
    return new Completion([this, task, indices = std::move(indices),
                           results = std::move(results)](TaskOutput&) mutable {
        return WaitForKeyResults(task, indices, std::move(results));
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchRemoveAsync(const std::vector<ObjectKey>& keys) {
    if (keys.empty()) {
        return new Completion([](TaskOutput&) {
            return std::vector<tl::expected<void, ErrorCode>>{};
        });
    }
    std::vector<kv::CacheKey> sdk_keys;
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    sdk_keys.reserve(keys.size());
    indices.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        sdk_keys.push_back(kv::CacheKey::FromString(keys[i]));
        indices.push_back(i);
    }
    kv::TaskId task = kv::kInvalidTaskId;
    auto submitted = kv_client_->DeleteAsync(sdk_keys, task, io_options_);
    if (!submitted.ok()) {
        LOG(WARNING) << "Failed to submit KVClient delete, keys="
                     << sdk_keys.size()
                     << ", status=" << static_cast<int>(submitted.code)
                     << ", error=" << submitted.message;
        return tl::unexpected(ToError(submitted));
    }
    return new Completion([this, task, indices = std::move(indices),
                           results = std::move(results)](TaskOutput&) mutable {
        return WaitForKeyResults(task, indices, std::move(results));
    });
}

}  // namespace mooncake
