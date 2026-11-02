#include "storage/kvclient_backend.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <glog/logging.h>
#include <iterator>
#include <limits>
#include <mutex>
#include <string_view>
#include <xxhash.h>
#include <utility>

#include "device/accelerator_registry.h"
#include "integer_parser.h"

namespace mooncake {

struct KvClientBackend::VllmKey {
    std::string_view model_name;
    std::string_view chunk_hash;
    // Retained fields form the object identity; ranks/layer select its offset.
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

std::pair<std::string_view, std::string_view> SplitKeyField(
    std::string_view field) {
    const auto separator = field.find(':');
    if (separator != std::string_view::npos)
        return {field.substr(0, separator), field.substr(separator + 1)};
    // Accept compact fields (pcp0/dcp0) as well as named fields (pcp:0/dcp:0).
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
    std::string_view key) const {
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
        uint64_t dimension_size = 0;
        if (name == "pp_rank") {
            rank = &parsed.pp_rank;
            present = &parsed.has_pp_rank;
            dimension_size = layout_->pp_size;
        } else if (name == "pcp") {
            rank = &parsed.pcp_rank;
            present = &parsed.has_pcp_rank;
            dimension_size = layout_->pcp_size;
        } else if (name == "dcp") {
            rank = &parsed.dcp_rank;
            present = &parsed.has_dcp_rank;
            dimension_size = layout_->dcp_size;
        } else if (name == "head_or_tp_rank" || name == "tp_rank") {
            rank = &parsed.head_or_tp_rank;
            present = &parsed.has_head_or_tp_rank;
            dimension_size = layout_->head_or_tp_size;
        }
        if (rank && dimension_size != 0) {
            const auto number = TryParseInteger<uint64_t>(value);
            if (!number || (*present && *rank != *number))
                return tl::unexpected(ErrorCode::INVALID_KEY);
            *rank = *number;
            *present = true;
        } else {
            // Keep unselected dimensions and unknown fields in the object key.
            if (present) *present = true;
            retained.push_back(field);
        }
        if (end == std::string_view::npos) break;
        key.remove_prefix(end + 1);
    }
    // Current LayerPoolKey omits PP rank and ends with @hash@<layer>.
    // At least model@hash@layer must remain after removing rank fields.
    constexpr size_t kMinLayerwiseKeyFields = 3;
    if (!parsed.has_pp_rank && retained.size() >= kMinLayerwiseKeyFields &&
        std::next(retained.rbegin())->find(':') == std::string_view::npos) {
        const auto last = retained.back();
        const bool numeric = std::all_of(last.begin(), last.end(), [](char c) {
            return c >= '0' && c <= '9';
        });
        if (numeric || last.front() == '-' || last.front() == '+') {
            const auto layer = TryParseInteger<uint64_t>(last);
            if (!layer) return tl::unexpected(ErrorCode::INVALID_KEY);
            parsed.layer = *layer;
            parsed.chunk_hash = *std::next(retained.rbegin());
            if (layout_->layer_count != 0) retained.pop_back();
        }
    }
    for (size_t i = 1; i < retained.size(); ++i) {
        if (!parsed.layer && retained[i].find(':') == std::string_view::npos)
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
    // Only selected dimensions constrain ranks and contribute to the offset.
    if ((layout_->pp_size != 0 &&
         ((!parsed.has_pp_rank && layout_->pp_size != 1) ||
          parsed.pp_rank >= layout_->pp_size)) ||
        (layout_->pcp_size != 0 &&
         ((!parsed.has_pcp_rank && layout_->pcp_size != 1) ||
          parsed.pcp_rank >= layout_->pcp_size)) ||
        (layout_->dcp_size != 0 &&
         ((!parsed.has_dcp_rank && layout_->dcp_size != 1) ||
          parsed.dcp_rank >= layout_->dcp_size)) ||
        (layout_->head_or_tp_size != 0 &&
         ((!parsed.has_head_or_tp_rank && layout_->head_or_tp_size != 1) ||
          parsed.head_or_tp_rank >= layout_->head_or_tp_size)) ||
        (layout_->layer_count != 0 && parsed.layer &&
         *parsed.layer >= layout_->layer_count))
        return tl::unexpected(ErrorCode::INVALID_KEY);
    return {};
}

tl::expected<kv::CacheKey, ErrorCode> KvClientBackend::EncodeKey(
    const ObjectKey& key, const TenantId& tenant_id, VllmKey* parsed_key) {
    ObjectKey identity = key;
    if (layout_) {
        // Reject invalid layout keys consistently for writes, reads and
        // deletes.
        auto parsed = ParseVllmKey(key);
        if (!parsed) return tl::unexpected(parsed.error());
        auto valid = ValidateKeyLayout(*parsed);
        if (!valid) return tl::unexpected(valid.error());
        identity = parsed->storage_key;
        // Reuse the parsed ranks when constructing transfer offsets.
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
    auto layout = ParseLayoutConfig(options);
    if (!layout) {
        LOG(ERROR) << "Invalid KVClient backend layout, error="
                   << static_cast<int>(layout.error());
        return tl::unexpected(layout.error());
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
    layout_ = *layout;
    return {};
}

tl::expected<std::optional<KvClientBackend::LayoutConfig>, ErrorCode>
KvClientBackend::ParseLayoutConfig(
    const std::unordered_map<std::string, std::string>& options) {
    LayoutConfig layout;
    const std::pair<const char*, uint64_t*> fields[] = {
        {"pp_rank", &layout.pp_size},
        {"pcp", &layout.pcp_size},
        {"dcp", &layout.dcp_size},
        {"layer", &layout.layer_count},
        {"head_or_tp_rank", &layout.head_or_tp_size},
    };
    // Reject obsolete layout options instead of silently changing object keys.
    for (const auto* name : {"pp_size", "pcp_size", "dcp_size", "layer_count",
                             "head_or_tp_size", "shard_bytes"}) {
        if (options.contains(name))
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    constexpr std::string_view kDimensionPrefix = "aggregation.dimensions.";
    bool has_dimensions = false;
    for (const auto& [name, text] : options) {
        uint64_t* destination = nullptr;
        if (name == "aggregation.shard_bytes") {
            destination = &layout.shard_bytes;
        } else if (name.starts_with(kDimensionPrefix)) {
            const auto dimension =
                std::string_view(name).substr(kDimensionPrefix.size());
            for (const auto& [field, value] : fields) {
                if (dimension == field) destination = value;
            }
            if (!destination) return tl::unexpected(ErrorCode::INVALID_PARAMS);
            has_dimensions = true;
        } else if (name == "aggregation" || name.starts_with("aggregation.")) {
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        } else {
            continue;
        }
        const auto value = TryParseInteger<uint64_t>(text);
        if (!value || *value == 0)
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        *destination = *value;
    }
    if (!has_dimensions) return std::nullopt;
    if (layout.shard_bytes == 0)
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    uint64_t object_capacity = layout.shard_bytes;
    for (const auto& [name, size] : fields) {
        if (*size == 0) continue;
        if (object_capacity > std::numeric_limits<uint64_t>::max() / *size)
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        object_capacity *= *size;
    }
    // SDK offsets are uint32; this is independent of storage capacity policy.
    if (object_capacity > uint64_t{std::numeric_limits<uint32_t>::max()} + 1)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    return std::optional<LayoutConfig>{layout};
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
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    std::vector<kv::KVBuffer> entries;
    std::vector<size_t> indices;
    std::vector<size_t> lengths;
    // Keep registrations stable until the SDK accepts the descriptors.
    std::shared_lock lock(memory_mutex_);
    for (size_t i = 0; i < keys.size(); ++i) {
        VllmKey parsed;
        auto key = EncodeKey(keys[i], tenant_id, &parsed);
        if (!key) {
            LOG(WARNING) << "Invalid KVClient load key, index=" << i
                         << ", key=" << keys[i]
                         << ", error=" << static_cast<int>(key.error());
            results[i] = tl::unexpected(key.error());
            continue;
        }
        const auto base_offset = ResolveBaseOffset(parsed);
        std::vector<kv::KVBuffer> object_entries;
        if (targets[i].empty()) {
            LOG(WARNING) << "Missing KVClient load targets for key=" << keys[i];
            results[i] = tl::unexpected(ErrorCode::INVALID_PARAMS);
            continue;
        }
        for (size_t j = 0; j < targets[i].size(); ++j) {
            const auto& slice = targets[i][j];
            // Range offsets are relative to this key's slot, not the SDK
            // object.
            const auto source_offset = source_offsets[i][j];
            if (source_offset > std::numeric_limits<uint32_t>::max() ||
                base_offset >
                    std::numeric_limits<uint32_t>::max() - source_offset) {
                LOG(WARNING)
                    << "KVClient load exceeds SDK offset limits for key="
                    << keys[i] << ", slice=" << j
                    << ", offset=" << source_offset
                    << ", base_offset=" << base_offset;
                results[i] = tl::unexpected(ErrorCode::NOT_SUPPORTED);
                break;
            }
            const auto object_offset = base_offset + source_offset;
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

uint64_t KvClientBackend::ResolveBaseOffset(const VllmKey& parsed) const {
    if (!layout_) return 0;
    // Fixed order: PP, PCP, DCP, layer, head/TP. Unselected axes are skipped.
    uint64_t index = 0;
    if (layout_->pp_size != 0) index = parsed.pp_rank;
    if (layout_->pcp_size != 0)
        index = index * layout_->pcp_size + parsed.pcp_rank;
    if (layout_->dcp_size != 0)
        index = index * layout_->dcp_size + parsed.dcp_rank;
    if (layout_->layer_count != 0 && parsed.layer)
        index = index * layout_->layer_count + *parsed.layer;
    if (layout_->head_or_tp_size != 0)
        index = index * layout_->head_or_tp_size + parsed.head_or_tp_rank;
    // Non-layerwise inputs already contain all layers and stay contiguous.
    const auto shard_stride_bytes =
        !parsed.layer && layout_->layer_count != 0
            ? layout_->layer_count * layout_->shard_bytes
            : layout_->shard_bytes;
    return index * shard_stride_bytes;
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
    uint64_t base_offset, std::vector<kv::KVBuffer>& entries) {
    if (slices.empty()) return tl::unexpected(ErrorCode::INVALID_PARAMS);
    // Preserve slice order even when the source addresses are noncontiguous.
    uint64_t logical_offset = 0;
    for (const auto& slice : slices) {
        auto region = FindRegisteredMemory(slice);
        if (!region) return tl::unexpected(region.error());
        const auto& registered = **region;
        const auto addr = reinterpret_cast<uintptr_t>(slice.ptr);
        uint64_t consumed = 0;
        while (consumed < slice.size) {
            if (logical_offset > std::numeric_limits<uint32_t>::max() ||
                base_offset >
                    std::numeric_limits<uint32_t>::max() - logical_offset)
                return tl::unexpected(ErrorCode::NOT_SUPPORTED);
            uint64_t object_offset = logical_offset;
            uint64_t length = slice.size - consumed;
            if (layout_) {
                object_offset += base_offset;
                // Split descriptors, never pad or copy source payload.
                length = std::min(length, uint64_t{0x00fffe00});
            }
            kv::KVBuffer entry{};
            entry.key = key;
            entry.buffer.region = registered.region;
            entry.buffer.region.addr = addr + consumed;
            entry.buffer.region.size = length;
            entry.buffer.handle = registered.handle;
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
        const auto begin = entries.size();
        results[i] =
            AppendStoreEntries(*key, batched_slices[i], base_offset, entries);
        if (!results[i]) {
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
        (status.ok() || status.code == kv::StatusCode::PARTIAL_FAILED) &&
        completed.entryStatus.size() == indices.size();
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
    auto submitted = kv_client_->StoreAsync(entries, task);
    memory_lock.unlock();
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        return WaitForKeyResults(task, indices, std::move(results));
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
    kv::TaskId task = kv::kInvalidTaskId;
    auto submitted = kv::Status::OK();
    if (!entries.empty())
        submitted = kv_client_->BatchStoreAsync(entries, task);
    memory_lock.unlock();
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        if (task == kv::kInvalidTaskId) return std::move(results);
        return WaitForKeyResults(task, indices, std::move(results));
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::RemoveAsync(const ObjectKey& key, const TenantId& tenant_id) {
    auto encoded = EncodeKey(key, tenant_id);
    if (!encoded) return tl::unexpected(encoded.error());
    kv::TaskId task;
    auto submitted = kv_client_->DeleteAsync({*encoded}, task);
    if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    return SaveTask([this, task] {
        return WaitForKeyResults(task, {0},
                                 std::vector<tl::expected<void, ErrorCode>>(1));
    });
}

tl::expected<ExternalStoreBackend::TaskHandle, ErrorCode>
KvClientBackend::BatchRemoveAsync(const std::vector<ObjectKey>& keys,
                                  const TenantId& tenant_id) {
    std::vector<kv::CacheKey> encoded_keys;
    std::vector<size_t> indices;
    std::vector<tl::expected<void, ErrorCode>> results(keys.size());
    encoded_keys.reserve(keys.size());
    indices.reserve(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        auto key = EncodeKey(keys[i], tenant_id);
        if (!key) {
            results[i] = tl::unexpected(key.error());
            continue;
        }
        encoded_keys.push_back(*key);
        indices.push_back(i);
    }
    kv::TaskId task = kv::kInvalidTaskId;
    if (!encoded_keys.empty()) {
        auto submitted = kv_client_->DeleteAsync(encoded_keys, task);
        if (!submitted.ok()) return tl::unexpected(ToError(submitted));
    }
    return SaveTask([this, task, indices = std::move(indices),
                     results = std::move(results)]() mutable {
        if (task == kv::kInvalidTaskId) return std::move(results);
        return WaitForKeyResults(task, indices, std::move(results));
    });
}

}  // namespace mooncake
