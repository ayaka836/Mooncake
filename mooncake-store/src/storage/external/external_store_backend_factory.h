#pragma once

#include <memory>
#include <string>

#include "storage/external/external_store_backend.h"

namespace mooncake {

/**
 * @brief Creates a backend instance; initialization is performed separately.
 * @param backend_type Configured backend name.
 * @return Backend instance, or NOT_SUPPORTED if the backend is unavailable.
 */
tl::expected<std::shared_ptr<ExternalStoreBackend>, ErrorCode>
CreateExternalStoreBackend(const std::string& backend_type);

}  // namespace mooncake
