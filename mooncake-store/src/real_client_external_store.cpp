#include "real_client.h"

#include <limits>
#include <utility>

#include "batch_read_fanout.h"

namespace mooncake {

std::vector<std::pair<size_t, tl::expected<bool, ErrorCode>>>
RealClient::check_external_get_candidates(
    const std::vector<ObjectKey>& keys,
    const std::vector<tl::expected<QueryResult, ErrorCode>>& query_results) {
    if (!client_->HasExternalStoreBackend()) return {};
    std::vector<ObjectKey> external_keys;
    std::vector<std::pair<size_t, tl::expected<bool, ErrorCode>>> results;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (!query_results[i] &&
            client_->ShouldGetFromExternalStore(query_results[i].error())) {
            external_keys.push_back(keys[i]);
            results.emplace_back(i, false);
        }
    }
    if (external_keys.empty()) return {};

    auto exists = client_->BatchExistsExternal(external_keys);
    for (size_t i = 0; i < external_keys.size(); ++i) {
        results[i].second = std::move(exists[i]);
    }
    return results;
}

tl::expected<void, ErrorCode>
RealClient::prepare_external_session_range_read_request(
    const ObjectKey& key, const std::vector<void*>& buffers,
    const std::vector<size_t>& sizes, const std::vector<size_t>& source_offsets,
    size_t result_index, ExternalReadRequests& requests) {
    size_t bytes = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        if ((!buffers[i] && sizes[i] != 0) ||
            sizes[i] >
                static_cast<size_t>(std::numeric_limits<int>::max()) - bytes) {
            return tl::unexpected(ErrorCode::INVALID_PARAMS);
        }
        bytes += sizes[i];
    }
    if (bytes == 0) return {};

    requests.keys.push_back(key);
    requests.targets.push_back(SlicesFromBuffers(buffers, sizes));
    requests.source_offsets.push_back(source_offsets);
    requests.result_indices.push_back(result_index);
    return {};
}

}  // namespace mooncake
