#include "client_service.h"

#include <algorithm>
#include <utility>
#include <glog/logging.h>
#include <ylt/struct_json/json_reader.h>

#include "environ.h"
#include "storage/external/external_store_backend_factory.h"

namespace mooncake {

tl::expected<void, ErrorCode>
Client::InitExternalStoreBackendFromEnvironment() {
    const auto backend_type =
        Environ::GetString("MC_EXTERNAL_STORE_BACKEND", "");
    if (backend_type.empty()) return {};
    std::unordered_map<std::string, std::string> options;
    try {
        struct_json::from_json(
            options, Environ::GetString("MC_EXTERNAL_STORE_OPTIONS", "{}"));
    } catch (const std::exception& error) {
        LOG(ERROR) << "Invalid MC_EXTERNAL_STORE_OPTIONS: " << error.what();
        return tl::unexpected(ErrorCode::INVALID_PARAMS);
    }
    auto backend = CreateExternalStoreBackend(backend_type);
    if (!backend) {
        LOG(ERROR) << "Failed to create external store backend: "
                   << toString(backend.error());
        return tl::unexpected(backend.error());
    }
    auto initialized = (*backend)->Init(options);
    if (!initialized) {
        LOG(ERROR) << "Failed to initialize external store backend: "
                   << toString(initialized.error());
        return tl::unexpected(initialized.error());
    }
    external_store_backend_ = std::move(*backend);
    return {};
}

bool Client::ShouldGetFromExternalStore(
    const tl::expected<QueryResult, ErrorCode>& query) const {
    if (!external_store_backend_) return false;
    if (!query) {
        return query.error() == ErrorCode::OBJECT_NOT_FOUND ||
               query.error() == ErrorCode::REPLICA_IS_NOT_READY;
    }
    return std::none_of(query->replicas.begin(), query->replicas.end(),
                        [](const auto& replica) {
                            return replica.status == ReplicaStatus::COMPLETE;
                        });
}

tl::expected<ExternalStoreTaskGuard, ErrorCode> Client::BatchGetExternalAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets,
    std::vector<size_t>* actual_sizes) {
    if (keys.empty()) return ExternalStoreTaskGuard{};
    if (!external_store_backend_)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    auto submitted = external_store_backend_->BatchGetAsync(
        keys, targets, TenantId(master_client_.tenant_id()), actual_sizes);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

tl::expected<ExternalStoreTaskGuard, ErrorCode>
Client::BatchGetExternalRangesAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets,
    const std::vector<std::vector<size_t>>& source_offsets,
    std::vector<size_t>* actual_sizes) {
    if (keys.empty()) return ExternalStoreTaskGuard{};
    if (!external_store_backend_)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    auto submitted = external_store_backend_->BatchGetRangesAsync(
        keys, targets, source_offsets, TenantId(master_client_.tenant_id()),
        actual_sizes);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

std::vector<tl::expected<int64_t, ErrorCode>> Client::WaitForExternalReads(
    tl::expected<ExternalStoreTaskGuard, ErrorCode>& submission,
    const std::vector<size_t>& actual_sizes, size_t key_count) {
    if (!submission) {
        return std::vector<tl::expected<int64_t, ErrorCode>>(
            key_count, tl::unexpected(submission.error()));
    }
    auto statuses = submission->Wait();
    std::vector<tl::expected<int64_t, ErrorCode>> results(statuses.size());
    for (size_t i = 0; i < results.size(); ++i) {
        if (statuses[i]) {
            results[i] = static_cast<int64_t>(actual_sizes[i]);
        } else {
            results[i] = tl::unexpected(statuses[i].error());
        }
    }
    return results;
}

tl::expected<ExternalStoreTaskGuard, ErrorCode>
Client::BatchExistsExternalAsync(const std::vector<ObjectKey>& keys,
                                 std::vector<uint8_t>* exists) {
    if (keys.empty()) return ExternalStoreTaskGuard{};
    if (!external_store_backend_)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    auto submitted = external_store_backend_->BatchExistsAsync(
        keys, TenantId(master_client_.tenant_id()), exists);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

std::vector<tl::expected<bool, ErrorCode>> Client::BatchExistsExternal(
    const std::vector<ObjectKey>& keys) {
    if (keys.empty()) return {};
    std::vector<uint8_t> exists;
    auto submitted = BatchExistsExternalAsync(keys, &exists);
    if (!submitted) {
        return std::vector<tl::expected<bool, ErrorCode>>(
            keys.size(), tl::unexpected(submitted.error()));
    }
    auto statuses = submitted->Wait();
    std::vector<tl::expected<bool, ErrorCode>> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (statuses[i]) {
            results[i] = exists[i] != 0;
        } else {
            results[i] = tl::unexpected(statuses[i].error());
        }
    }
    return results;
}

}  // namespace mooncake
