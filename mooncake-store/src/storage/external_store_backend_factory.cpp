#include "storage/external_store_backend_factory.h"

#ifdef STORE_USE_KVCLIENT
#include "storage/kvclient_backend.h"
#endif

namespace mooncake {

tl::expected<std::shared_ptr<ExternalStoreBackend>, ErrorCode>
CreateExternalStoreBackend([[maybe_unused]] const std::string& backend_type) {
#ifdef STORE_USE_KVCLIENT
    if (backend_type == "kvclient") {
        return std::shared_ptr<ExternalStoreBackend>(
            std::make_shared<KvClientBackend>());
    }
#endif
    return tl::unexpected(ErrorCode::NOT_SUPPORTED);
}

}  // namespace mooncake
