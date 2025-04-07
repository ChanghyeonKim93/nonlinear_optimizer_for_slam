#ifndef NONLINEAR_OPTIMIZER_UNORDERED_BIMAP_H_
#define NONLINEAR_OPTIMIZER_UNORDERED_BIMAP_H_

#include <iostream>
#include <unordered_map>

namespace nonlinear_optimizer {

template <typename kTypeKey, typename kTypeValue>
class UnorderedBimap {
 public:
  UnorderedBimap() {}

  ~UnorderedBimap() {}

  inline bool IsKeyExist(const kTypeKey& key) const {
    return key_to_val_map_.find(key) != key_to_val_map_.end();
  }

  inline bool IsValueExist(const kTypeValue& val) const {
    return val_to_key_map_.find(val) != val_to_key_map_.end();
  }

  void Insert(const kTypeKey& key, const kTypeValue& val) {
    if (IsKeyExist(key) || IsValueExist(val)) {
      std::cerr << "Either key or value already exists!\n";
      return;
    }
    key_to_val_map_.insert({key, val});
    val_to_key_map_.insert({val, key});
  }

  void Insert(const std::pair<kTypeKey, kTypeValue>& pair) {
    if (IsKeyExist(pair.first) || IsValueExist(pair.second)) {
      std::cerr << "Either key or value already exists!\n";
      return;
    }
    key_to_val_map_.insert({pair.first, pair.second});
    val_to_key_map_.insert({pair.second, pair.first});
  }

  const kTypeKey& GetKey(const kTypeValue& val) const {
    return val_to_key_map_.at(val);
  }

  const kTypeValue& GetValue(const kTypeKey& key) const {
    return key_to_val_map_.at(key);
  }

  void DeleteByKey(const kTypeKey& key) {
    if (key_to_val_map_.find(key) == key_to_val_map_.end()) return;
    const auto& val = key_to_val_map_.at(key);
    val_to_key_map_.erase(val);
    key_to_val_map_.erase(key);
  }

  void DeleteByValue(const kTypeValue& val) {
    if (val_to_key_map_.find(val) == val_to_key_map_.end()) return;
    const auto& key = val_to_key_map_.at(key);
    key_to_val_map_.erase(key);
    val_to_key_map_.erase(val);
  }

  void Clear() {
    key_to_val_map_.clear();
    val_to_key_map_.clear();
  }

 private:
  std::unordered_map<kTypeKey, kTypeValue> key_to_val_map_;
  std::unordered_map<kTypeValue, kTypeKey> val_to_key_map_;
};

}  // namespace nonlinear_optimizer

#endif  // NONLINEAR_OPTIMIZER_UNORDERED_BIMAP_H_
