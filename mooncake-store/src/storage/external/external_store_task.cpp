#include "storage/external/external_store_task.h"

#include <exception>
#include <utility>

#include <glog/logging.h>

namespace mooncake {

ExternalStoreTaskGuard::ExternalStoreTaskGuard(
    ExternalStoreBackend* backend, ExternalStoreBackend::TaskHandle task)
    : backend_(backend), task_(task) {}

ExternalStoreTaskGuard::~ExternalStoreTaskGuard() {
    if (!task_) return;
    try {
        ExternalStoreBackend::TaskOutput output;
        (void)Wait(output);
    } catch (const std::exception& e) {
        LOG(ERROR) << "External store task threw during cleanup: " << e.what();
    } catch (...) {
        LOG(ERROR) << "External store task threw during cleanup";
    }
}

ExternalStoreTaskGuard::ExternalStoreTaskGuard(
    ExternalStoreTaskGuard&& other) noexcept
    : backend_(std::exchange(other.backend_, nullptr)),
      task_(std::exchange(other.task_, std::nullopt)) {}

ExternalStoreTaskGuard& ExternalStoreTaskGuard::operator=(
    ExternalStoreTaskGuard&& other) {
    if (this != &other) {
        if (task_) {
            ExternalStoreBackend::TaskOutput output;
            (void)Wait(output);
        }
        backend_ = std::exchange(other.backend_, nullptr);
        task_ = std::exchange(other.task_, std::nullopt);
    }
    return *this;
}

std::vector<tl::expected<void, ErrorCode>> ExternalStoreTaskGuard::Wait(
    ExternalStoreBackend::TaskOutput& output) {
    auto task = std::exchange(task_, std::nullopt);
    if (!task) {
        output = {};
        return {};
    }
    return backend_->Wait(*task, output);
}

}  // namespace mooncake
