#pragma once

#include <optional>

#include "storage/external/external_store_backend.h"

namespace mooncake {

/**
 * @brief Owns an accepted backend task and waits on destruction if still
 * pending.
 */
class ExternalStoreTaskGuard {
   public:
    ExternalStoreTaskGuard() = default;

    /**
     * @brief Takes ownership of an accepted task without owning its backend.
     * @param backend Backend that stays usable until the task is consumed.
     * @param task Handle consumed by Wait or destruction.
     */
    ExternalStoreTaskGuard(ExternalStoreBackend* backend,
                           ExternalStoreBackend::TaskHandle task);
    ~ExternalStoreTaskGuard();

    ExternalStoreTaskGuard(const ExternalStoreTaskGuard&) = delete;
    ExternalStoreTaskGuard& operator=(const ExternalStoreTaskGuard&) = delete;
    ExternalStoreTaskGuard(ExternalStoreTaskGuard&& other) noexcept;
    ExternalStoreTaskGuard& operator=(ExternalStoreTaskGuard&& other);

    explicit operator bool() const { return task_.has_value(); }

    /**
     * @brief Waits for completion and consumes the task.
     * @param output Operation results supplied to the backend Wait.
     * @return Backend statuses, or empty statuses and output for an empty or
     * moved-from guard. Such guards perform no backend work.
     */
    std::vector<tl::expected<void, ErrorCode>> Wait(
        ExternalStoreBackend::TaskOutput& output);

   private:
    ExternalStoreBackend* backend_ = nullptr;
    std::optional<ExternalStoreBackend::TaskHandle> task_;
};

}  // namespace mooncake
