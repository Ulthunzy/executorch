#pragma once

#include <memory>
#include <string>

std::string unpack_model_package(
    const std::string& filename,
    const std::string& cache_directory = "");
void link_model_libraries(
    const std::string& model_root,
    const std::string& library_directory);

class ModelPackageLock;
std::shared_ptr<ModelPackageLock> lock_package_model(
    const std::string& model_root);
