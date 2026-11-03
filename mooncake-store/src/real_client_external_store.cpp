#include "real_client.h"

#include <limits>
#include <utility>

namespace mooncake {

tl::expected<void, ErrorCode> RealClient::append_external_range_read_request(
    ExternalReadRequests& requests, const ObjectKey& key,
    std::vector<Slice> targets, const std::vector<size_t>& source_offsets,
    size_t result_index) {
    size_t bytes = 0;
    for (const auto& target : targets) {
        if ((!target.ptr && target.size != 0) ||
            target.size >
                static_cast<size_t>(std::numeric_limits<int>::max()) - bytes) {
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
        bytes += target.size;
    }
    if (bytes != 0) {
        requests.keys.push_back(key);
        requests.targets.push_back(std::move(targets));
        requests.source_offsets.push_back(source_offsets);
        requests.result_indices.push_back(result_index);
    }
    return {};
}

std::unordered_map<size_t, tl::expected<void, ErrorCode>>
RealClient::check_external_get_candidates(
    const std::vector<ObjectKey>& keys,
    const std::vector<tl::expected<QueryResult, ErrorCode>>& query_results) {
    std::vector<ObjectKey> external_keys;
    std::vector<size_t> indices;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (client_->ShouldGetFromExternalStore(query_results[i])) {
            external_keys.push_back(keys[i]);
            indices.push_back(i);
        }
    }

    const auto exists = client_->BatchExistsExternal(external_keys);
    std::unordered_map<size_t, tl::expected<void, ErrorCode>> results;
    for (size_t i = 0; i < external_keys.size(); ++i) {
        if (exists[i] && *exists[i]) {
            results.emplace(indices[i], tl::expected<void, ErrorCode>{});
        } else {
            results.emplace(
                indices[i],
                tl::unexpected(exists[i] ? ErrorCode::OBJECT_NOT_FOUND
                                         : exists[i].error()));
        }
    }
    return results;
}

}  // namespace mooncake
