#include "client_service.h"

#include <utility>
#include <glog/logging.h>
#include <ylt/struct_json/json_reader.h>

#include "environ.h"
#include "environment_variables.h"
#include "storage/external/external_store_backend_factory.h"

namespace mooncake {

tl::expected<void, ErrorCode>
Client::InitExternalStoreBackendFromEnvironment() {
    using Variables = ExternalStoreEnvironmentVariables;
    const auto backend_type =
        Environ::ReadOr(Variables::MC_EXTERNAL_STORE_BACKEND, std::string{});
    if (backend_type.empty()) return {};
    std::unordered_map<std::string, std::string> options;
    try {
        struct_json::from_json(
            options, Environ::ReadOr(Variables::MC_EXTERNAL_STORE_OPTIONS,
                                     std::string{"{}"}));
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
    auto initialized =
        (*backend)->Init(options, TenantId(master_client_.tenant_id()));
    if (!initialized) {
        LOG(ERROR) << "Failed to initialize external store backend: "
                   << toString(initialized.error());
        return tl::unexpected(initialized.error());
    }
    external_store_backend_ = std::move(*backend);
    LOG(INFO) << "External store backend initialized successfully: "
              << backend_type;
    return {};
}

bool Client::ShouldGetFromExternalStore(ErrorCode query_error) const {
    return external_store_backend_ &&
           (query_error == ErrorCode::OBJECT_NOT_FOUND ||
            query_error == ErrorCode::REPLICA_IS_NOT_READY);
}

tl::expected<ExternalStoreTaskGuard, ErrorCode> Client::BatchGetExternalAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets) {
    if (keys.empty()) return ExternalStoreTaskGuard{};
    if (!external_store_backend_)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    auto submitted = external_store_backend_->BatchGetAsync(keys, targets);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

tl::expected<ExternalStoreTaskGuard, ErrorCode>
Client::BatchGetExternalRangesAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& targets,
    const std::vector<std::vector<size_t>>& source_offsets) {
    if (keys.empty()) return ExternalStoreTaskGuard{};
    if (!external_store_backend_)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    auto submitted = external_store_backend_->BatchGetRangesAsync(
        keys, targets, source_offsets);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

tl::expected<ExternalStoreTaskGuard, ErrorCode>
Client::BatchExistsExternalAsync(const std::vector<ObjectKey>& keys) {
    if (keys.empty()) return ExternalStoreTaskGuard{};
    if (!external_store_backend_)
        return tl::unexpected(ErrorCode::NOT_SUPPORTED);
    auto submitted = external_store_backend_->BatchExistsAsync(keys);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

std::vector<tl::expected<bool, ErrorCode>> Client::BatchExistsExternal(
    const std::vector<ObjectKey>& keys) {
    if (keys.empty()) return {};
    auto submitted = BatchExistsExternalAsync(keys);
    if (!submitted) {
        return std::vector<tl::expected<bool, ErrorCode>>(
            keys.size(), tl::unexpected(submitted.error()));
    }
    ExternalStoreBackend::TaskOutput output;
    auto statuses = submitted->Wait(output);
    std::vector<tl::expected<bool, ErrorCode>> results(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        if (statuses[i]) {
            results[i] = output.exists[i] != 0;
        } else {
            results[i] = tl::unexpected(statuses[i].error());
        }
    }
    return results;
}

tl::expected<ExternalStoreTaskGuard, ErrorCode> Client::BatchPutExternalAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& sources) {
    if (keys.empty() || !external_store_backend_)
        return ExternalStoreTaskGuard{};
    auto submitted = external_store_backend_->BatchPutAsync(keys, sources);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

tl::expected<ExternalStoreTaskGuard, ErrorCode>
Client::BatchPutExternalRangesAsync(
    const std::vector<ObjectKey>& keys,
    const std::vector<std::vector<Slice>>& sources,
    const std::vector<std::vector<size_t>>& target_offsets) {
    if (keys.empty() || !external_store_backend_)
        return ExternalStoreTaskGuard{};
    auto submitted = external_store_backend_->BatchPutRangesAsync(
        keys, sources, target_offsets);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

tl::expected<ExternalStoreTaskGuard, ErrorCode>
Client::BatchRemoveExternalAsync(const std::vector<ObjectKey>& keys) {
    if (keys.empty() || !external_store_backend_)
        return ExternalStoreTaskGuard{};
    auto submitted = external_store_backend_->BatchRemoveAsync(keys);
    if (!submitted) return tl::unexpected(submitted.error());
    return ExternalStoreTaskGuard(external_store_backend_.get(), *submitted);
}

}  // namespace mooncake
