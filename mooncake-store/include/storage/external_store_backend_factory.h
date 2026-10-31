#pragma once

#include <memory>
#include <string>

#include "storage/external_store_backend.h"

namespace mooncake {

tl::expected<std::shared_ptr<ExternalStoreBackend>, ErrorCode>
CreateExternalStoreBackend(const std::string& backend_type);

}  // namespace mooncake
