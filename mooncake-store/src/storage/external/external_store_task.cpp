#include "storage/external/external_store_task.h"

#include <utility>

namespace mooncake {

ExternalStoreTaskGuard::ExternalStoreTaskGuard(
    ExternalStoreBackend* backend, ExternalStoreBackend::TaskHandle task)
    : backend_(backend), task_(task) {}

ExternalStoreTaskGuard::~ExternalStoreTaskGuard() {
    if (task_) (void)Wait();
}

ExternalStoreTaskGuard::ExternalStoreTaskGuard(
    ExternalStoreTaskGuard&& other) noexcept
    : backend_(std::exchange(other.backend_, nullptr)),
      task_(std::exchange(other.task_, std::nullopt)) {}

ExternalStoreTaskGuard& ExternalStoreTaskGuard::operator=(
    ExternalStoreTaskGuard&& other) {
    if (this != &other) {
        if (task_) (void)Wait();
        backend_ = std::exchange(other.backend_, nullptr);
        task_ = std::exchange(other.task_, std::nullopt);
    }
    return *this;
}

std::vector<tl::expected<void, ErrorCode>> ExternalStoreTaskGuard::Wait() {
    if (!task_) return {};
    auto statuses = backend_->Wait(*task_);
    task_.reset();
    return statuses;
}

}  // namespace mooncake
