#include "storage/kvclient_backend.h"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <string_view>
#include <xxhash.h>
#include <utility>

#include "device/accelerator_registry.h"

namespace mooncake {

struct KvClientBackend::VllmKey {
    std::string_view model_name;
    std::string_view chunk_hash;
    std::string storage_key;
    uint64_t pp_rank = 0;
    uint64_t pcp_rank = 0;
    uint64_t dcp_rank = 0;
    uint64_t head_or_tp_rank = 0;
    std::optional<uint64_t> layer;
    bool has_pp_rank = false;
    bool has_pcp_rank = false;
    bool has_dcp_rank = false;
    bool has_head_or_tp_rank = false;
};

namespace {

bool ParseRankValue(std::string_view text, uint64_t& value) {
    const auto [end, error] =
        std::from_chars(text.data(), text.data() + text.size(), value);
    return error == std::errc{} && end == text.data() + text.size();
}

std::pair<std::string_view, std::string_view> SplitKeyField(
    std::string_view field) {
    const auto separator = field.find(':');
    if (separator != std::string_view::npos)
        return {field.substr(0, separator), field.substr(separator + 1)};
    for (std::string_view name : {"pcp", "dcp"}) {
        if (!field.starts_with(name)) continue;
        const auto value = field.substr(name.size());
        if (value.empty() || (value.front() >= '0' && value.front() <= '9') ||
            value.front() == '-' || value.front() == '+')
            return {name, value};
    }
    return {field, {}};
}

}  // namespace

tl::expected<KvClientBackend::VllmKey, ErrorCode> KvClientBackend::ParseVllmKey(
    std::string_view key) {
    VllmKey parsed;
    const auto first = key.find('@');
    if (first == std::string_view::npos || first == 0)
        return tl::unexpected(ErrorCode::INVALID_KEY);
    parsed.model_name = key.substr(0, first);
    std::vector<std::string_view> retained{parsed.model_name};
    key.remove_prefix(first + 1);
    while (true) {
        const auto end = key.find('@');
        const auto field = key.substr(0, end);
        if (field.empty()) return tl::unexpected(ErrorCode::INVALID_KEY);
        const auto [name, value] = SplitKeyField(field);
        uint64_t* rank = nullptr;
        bool* present = nullptr;
        if (name == "pp_rank") {
            rank = &parsed.pp_rank;
            present = &parsed.has_pp_rank;
        } else if (name == "pcp") {
            rank = &parsed.pcp_rank;
            present = &parsed.has_pcp_rank;
        } else if (name == "dcp") {
            rank = &parsed.dcp_rank;
            present = &parsed.has_dcp_rank;
        } else if (name == "head_or_tp_rank" || name == "tp_rank") {
            rank = &parsed.head_or_tp_rank;
            present = &parsed.has_head_or_tp_rank;
        }
        if (rank) {
            uint64_t number;
            if (!ParseRankValue(value, number) || (*present && *rank != number))
                return tl::unexpected(ErrorCode::INVALID_KEY);
            *rank = number;
            *present = true;
        } else {
            // Preserve unknown fields, values and order in the object identity.
            retained.push_back(field);
        }
        if (end == std::string_view::npos) break;
        key.remove_prefix(end + 1);
    }
    // Current LayerPoolKey omits PP rank and ends with @hash@<layer>.
    if (!parsed.has_pp_rank && retained.size() >= 3 &&
        retained[retained.size() - 2].find(':') == std::string_view::npos) {
        const auto last = retained.back();
        const bool numeric = std::all_of(last.begin(), last.end(), [](char c) {
            return c >= '0' && c <= '9';
        });
        if (numeric || last.front() == '-' || last.front() == '+') {
            uint64_t layer;
            if (!ParseRankValue(last, layer))
                return tl::unexpected(ErrorCode::INVALID_KEY);
            parsed.layer = layer;
            retained.pop_back();
        }
    }
    for (size_t i = 1; i < retained.size(); ++i) {
        if (retained[i].find(':') == std::string_view::npos)
            parsed.chunk_hash = retained[i];
    }
    if (parsed.chunk_hash.empty())
        return tl::unexpected(ErrorCode::INVALID_KEY);
    for (const auto field : retained) {
        if (!parsed.storage_key.empty()) parsed.storage_key.push_back('@');
        parsed.storage_key.append(field);
    }
    return parsed;
}

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

tl::expected<void, ErrorCode> KvClientBackend::ValidateKeyLayout(
    const VllmKey& parsed) const {
    if ((!parsed.has_pp_rank && layout_->pp_size != 1) ||
        (!parsed.has_pcp_rank && layout_->pcp_size != 1) ||
        (!parsed.has_dcp_rank && layout_->dcp_size != 1) ||
        (!parsed.has_head_or_tp_rank && layout_->head_or_tp_size != 1) ||
        parsed.pp_rank >= layout_->pp_size ||
        parsed.pcp_rank >= layout_->pcp_size ||
        parsed.dcp_rank >= layout_->dcp_size ||
        parsed.head_or_tp_rank >= layout_->head_or_tp_size ||
        (parsed.layer && *parsed.layer >= layout_->layer_count))
        return tl::unexpected(ErrorCode::INVALID_KEY);
    return {};
}

tl::expected<kv::CacheKey, ErrorCode> KvClientBackend::EncodeKey(
    const ObjectKey& key, const TenantId& tenant_id, VllmKey* parsed_key) {
    ObjectKey identity = key;
    if (layout_) {
        auto parsed = ParseVllmKey(key);
        if (!parsed) return tl::unexpected(parsed.error());
        auto valid = ValidateKeyLayout(*parsed);
        if (!valid) return tl::unexpected(valid.error());
        identity = parsed->storage_key;
        if (parsed_key) *parsed_key = std::move(*parsed);
    }
    const auto scoped_key = tenant_id.MakeScopedKey(identity);
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
    auto layout = ParseLayoutConfig(options);
    if (!layout) return tl::unexpected(layout.error());
    sdk_ = kv::CreateKvClient();
    auto status = sdk_->Init(config->second);
    if (!status.ok()) return tl::unexpected(ToError(status));
    layout_ = *layout;
    return {};
}

tl::expected<std::optional<KvClientBackend::LayoutConfig>, ErrorCode>
KvClientBackend::ParseLayoutConfig(
    const std::unordered_map<std::string, std::string>& options) {
    LayoutConfig layout;
    const std::pair<const char*, uint64_t*> fields[] = {
        {"pp_size", &layout.pp_size},
        {"pcp_size", &layout.pcp_size},
        {"dcp_size", &layout.dcp_size},
        {"layer_count", &layout.layer_count},
        {"head_or_tp_size", &layout.head_or_tp_size},
        {"shard_bytes", &layout.shard_bytes},
    };
    // Existing SDK-only configurations remain valid. A layout, when supplied,
    // must specify every dimension shared by all writers of an object.
    const bool has_layout = std::any_of(
        std::begin(fields), std::end(fields),
        [&](const auto& field) { return options.contains(field.first); });
    if (has_layout) {
        uint64_t object_capacity = 1;
        for (const auto& [name, value] : fields) {
            const auto it = options.find(name);
            if (it == options.end())
                return tl::unexpected(ErrorCode::INVALID_PARAMS);
            const auto& text = it->second;
            const auto [end, error] =
                std::from_chars(text.data(), text.data() + text.size(), *value);
            if (error != std::errc{} || end != text.data() + text.size() ||
                *value == 0 ||
                object_capacity > std::numeric_limits<uint64_t>::max() / *value)
                return tl::unexpected(ErrorCode::INVALID_PARAMS);
            object_capacity *= *value;
        }
        if (layout.shard_bytes % 512 != 0)
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        if (layout.pp_size != 1 ||
            object_capacity >
                uint64_t{std::numeric_limits<uint32_t>::max()} + 1)
            return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    }
    return has_layout ? std::optional<LayoutConfig>{layout} : std::nullopt;
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
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    sdk_keys.reserve(keys.size());
    indices.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        auto key = EncodeKey(keys[i], tenant_id);
        if (!key) {
            results[i] = tl::unexpected(key.error());
            continue;
        }
        sdk_keys.push_back(*key);
        indices.push_back(i);
    }
    if (sdk_keys.empty()) {
        return SaveTask([exists, results = std::move(results)]() mutable {
            exists->assign(results.size(), 0);
            return std::move(results);
        });
    }
    kv::TaskId task;
    auto submitted = sdk_->QueryAsync(sdk_keys, task);
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task, exists, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        auto completed = WaitForCompletion(task);
        exists->assign(results.size(), 0);
        for (size_t j = 0; j < indices.size(); ++j) {
            const auto index = indices[j];
            const auto& status = completed.entryStatus.empty()
                                     ? completed.status
                                     : completed.entryStatus[j];
            if (status.code == kv::StatusCode::ENTRY_KEY_EXIST) {
                (*exists)[index] = 1;
            } else if (ToError(status) == ErrorCode::OBJECT_NOT_FOUND &&
                       !status.ok()) {
                (*exists)[index] = 0;
            } else if (!status.ok()) {
                results[index] = tl::unexpected(ToError(status));
            } else {
                (*exists)[index] = completed.queryResult->exists[j] != 0;
            }
        }
        return std::move(results);
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
        VllmKey parsed;
        auto key = EncodeKey(keys[i], tenant_id, &parsed);
        if (!key) {
            results[i] = tl::unexpected(key.error());
            continue;
        }
        const auto base_offset = ResolveBaseOffset(parsed);
        const auto capacity =
            layout_
                ? (parsed.layer ? layout_->shard_bytes
                                : layout_->layer_count * layout_->shard_bytes)
                : 0;
        std::vector<kv::KVBuffer> object_entries;
        if (targets[i].empty()) {
            results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
            continue;
        }
        for (size_t j = 0; j < targets[i].size(); ++j) {
            const auto& slice = targets[i][j];
            uint64_t object_offset = source_offsets[i][j];
            if (layout_) {
                if (object_offset > capacity ||
                    slice.size > capacity - object_offset) {
                    results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
                    break;
                }
                object_offset += base_offset;
            }
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
            entry.key = *key;
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

uint64_t KvClientBackend::ResolveBaseOffset(const VllmKey& parsed) const {
    if (!layout_) return 0;
    const auto rank = (parsed.pp_rank * layout_->pcp_size + parsed.pcp_rank) *
                          layout_->dcp_size +
                      parsed.dcp_rank;
    if (parsed.layer) {
        const auto base_offset =
            ((rank * layout_->layer_count + *parsed.layer) *
                 layout_->head_or_tp_size +
             parsed.head_or_tp_rank) *
            layout_->shard_bytes;
        return base_offset;
    }
    // A non-layerwise key occupies one contiguous slot for all its slices.
    const auto capacity = layout_->layer_count * layout_->shard_bytes;
    const auto base_offset =
        (rank * layout_->head_or_tp_size + parsed.head_or_tp_rank) * capacity;
    return base_offset;
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
    uint64_t base_offset, uint64_t capacity,
    std::vector<kv::KVBuffer>& entries) {
    if (slices.empty()) return tl::unexpected(ErrorCode::INVALID_PARAMS);
    uint64_t logical_offset = 0;
    for (const auto& slice : slices) {
        auto registered = FindRegisteredMemory(slice);
        if (!registered) return tl::unexpected(registered.error());
        const auto addr = reinterpret_cast<uintptr_t>(slice.ptr);
        uint64_t consumed = 0;
        while (consumed < slice.size) {
            uint64_t object_offset = logical_offset;
            uint64_t length = slice.size - consumed;
            if (layout_) {
                if (logical_offset > capacity ||
                    length > capacity - logical_offset)
                    return tl::unexpected(ErrorCode::INVALID_PARAMS);
                object_offset += base_offset;
                // Split descriptors, never pad or copy source payload.
                length = std::min(length, uint64_t{0x00fffe00});
                if (object_offset % 512 != 0 || length % 512 != 0)
                    return tl::unexpected(ErrorCode::INVALID_PARAMS);
            }
            kv::KVBuffer entry{};
            entry.key = key;
            entry.buffer.region = (*registered)->region;
            entry.buffer.region.addr = addr + consumed;
            entry.buffer.region.size = length;
            entry.buffer.handle = (*registered)->handle;
            entry.offset = static_cast<uint32_t>(object_offset);
            entries.push_back(entry);
            consumed += length;
            logical_offset += length;
        }
    }
    return {};
}

std::vector<kv::KVBuffer> KvClientBackend::BuildStoreEntries(
    const TenantId& tenant_id, const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& batched_slices,
    std::vector<size_t>& indices,
    std::vector<tl::expected<void, ErrorCode>>& results) {
    // The caller holds a shared memory lock while resolving registrations.
    std::vector<kv::KVBuffer> entries;
    results.resize(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        VllmKey parsed;
        auto key = EncodeKey(keys[i], tenant_id, &parsed);
        if (!key) {
            results[i] = tl::unexpected(key.error());
            continue;
        }
        const auto base_offset = ResolveBaseOffset(parsed);
        const auto capacity =
            layout_
                ? (parsed.layer ? layout_->shard_bytes
                                : layout_->layer_count * layout_->shard_bytes)
                : 0;
        const auto begin = entries.size();
        results[i] = AppendStoreEntries(*key, batched_slices[i], base_offset,
                                        capacity, entries);
        if (!results[i]) {
            entries.resize(begin);
            continue;
        }
        indices.insert(indices.end(), entries.size() - begin, i);
    }
    return entries;
}

std::vector<tl::expected<void, ErrorCode>> KvClientBackend::WaitForWrite(
    kv::TaskId task, const std::vector<size_t>& indices,
    std::vector<tl::expected<void, ErrorCode>> results) {
    const auto completed = WaitForCompletion(task);
    const auto& status = completed.status;
    const bool has_entries =
        (status.ok() || status.code == kv::StatusCode::PARTIAL_FAILED) &&
        completed.entryStatus.size() == indices.size();
    for (size_t j = 0; j < indices.size(); ++j) {
        const auto error =
            ToError(has_entries ? completed.entryStatus[j] : status);
        auto& item = results[indices[j]];
        if (item && error != ErrorCode::OK) item = tl::unexpected(error);
    }
    return results;
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::PutAsync(const ObjectKey& key,
                          const std::vector<Slice>& slices,
                          const TenantId& tenant_id) {
    std::shared_lock memory_lock(memory_mutex_);
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results;
    auto entries =
        BuildStoreEntries(tenant_id, {key}, {slices}, indices, results);
    if (!results.front()) return tl::unexpected(results.front().error());
    kv::TaskId task;
    auto submitted = sdk_->StoreAsync(entries, task);
    memory_lock.unlock();
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        return WaitForWrite(task, indices, std::move(results));
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchPutAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& batched_slices,
    const TenantId& tenant_id) {
    if (keys.size() != batched_slices.size())
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    std::shared_lock memory_lock(memory_mutex_);
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results;
    auto entries =
        BuildStoreEntries(tenant_id, keys, batched_slices, indices, results);
    if (entries.empty()) {
        memory_lock.unlock();
        return SaveTask([results = std::move(results)]() mutable {
            return std::move(results);
        });
    }
    kv::TaskId task;
    auto submitted = sdk_->BatchStoreAsync(entries, task);
    memory_lock.unlock();
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        return WaitForWrite(task, indices, std::move(results));
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::RemoveAsync(const ObjectKey& key, const TenantId& tenant_id) {
    auto encoded = EncodeKey(key, tenant_id);
    if (!encoded) return tl::unexpected(encoded.error());
    kv::TaskId task;
    auto submitted = sdk_->DeleteAsync({*encoded}, task);
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task] {
        return WaitForWrite(task, {0},
                            std::vector<tl::expected<void, ErrorCode>>(1));
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchRemoveAsync(const std::vector<ObjectKey>& keys,
                                  const TenantId& tenant_id) {
    if (keys.empty()) {
        return SaveTask(
            [] { return std::vector<tl::expected<void, ErrorCode>>{}; });
    }
    std::vector<kv::CacheKey> sdk_keys;
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    sdk_keys.reserve(keys.size());
    indices.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        auto key = EncodeKey(keys[i], tenant_id);
        if (!key) {
            results[i] = tl::unexpected(key.error());
            continue;
        }
        sdk_keys.push_back(*key);
        indices.push_back(i);
    }
    if (sdk_keys.empty()) {
        return SaveTask([results = std::move(results)]() mutable {
            return std::move(results);
        });
    }
    kv::TaskId task;
    auto submitted = sdk_->DeleteAsync(sdk_keys, task);
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        return WaitForWrite(task, indices, std::move(results));
    });
}

}  // namespace mooncake
