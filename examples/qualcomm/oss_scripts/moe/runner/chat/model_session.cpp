#include "model_session.h"
#include <map>
#include <mutex>
#include <stdexcept>

namespace moe_chat {
namespace {
std::mutex registry_mutex;
std::map<std::string, SessionFactory> factories;
}

void register_model_session(const std::string& family, SessionFactory factory) {
  if (family.empty() || !factory)
    throw std::invalid_argument("A model family and session factory are required");
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto entry = factories.emplace(family, factory);
  if (!entry.second && entry.first->second != factory)
    throw std::runtime_error("Model session already registered: " + family);
}

std::unique_ptr<ModelSession> create_model_session(
    const std::string& family, Options options, Client* client) {
  SessionFactory factory;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = factories.find(family);
    if (found == factories.end())
      throw std::invalid_argument("Unsupported model session: " + family);
    factory = found->second;
  }
  options.model_family = family;
  auto session = factory(std::move(options), client);
  if (!session)
    throw std::runtime_error("Model session factory returned null: " + family);
  return session;
}
} // namespace moe_chat
