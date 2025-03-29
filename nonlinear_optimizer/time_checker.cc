#include "nonlinear_optimizer/time_checker.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nonlinear_optimizer {

double TimeCheckerManager::min_time_threshold_{1e-7};
std::string TimeCheckerManager::save_file_name_{""};

TimeCheckerManager* TimeCheckerManager::GetSingletonInstance() {
  static TimeCheckerManager time_checker_manager;
  return &time_checker_manager;
}

TimeChecker::TimeChecker(const std::string& file_name, const std::string& function_name)
    : name_(file_name + "/" + function_name), start_time_(std::chrono::high_resolution_clock::now()) {}

TimeChecker::~TimeChecker() {
  std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
  const double time_in_msec = (end_time - start_time_).count() * 1e-6;
  TimeCheckerManager::GetSingletonInstance()->RegisterTime(name_, time_in_msec);
}

TimeCheckerManager::TimeCheckerManager() {}

void TimeCheckerManager::RegisterTime(const std::string& time_checker_name, const double time) {
  if (time_list_of_functions_.find(time_checker_name) == time_list_of_functions_.end()) {
    time_list_of_functions_.insert({time_checker_name, std::vector<double>()});
    time_list_of_functions_.at(time_checker_name).reserve(1000000);
  }
  time_list_of_functions_.at(time_checker_name).push_back(time);
}

void TimeCheckerManager::SetMinTimeThreshold(const double min_time_threshold) {
  min_time_threshold_ = min_time_threshold;
}

void TimeCheckerManager::SaveFile(const std::string& file_name) { save_file_name_ = file_name; }

TimeCheckerManager::~TimeCheckerManager() {
  std::stringstream ss;
  ss << "------------ Time Analysis ------------" << std::endl;
  ss.precision(6);
  ss.setf(std::ios::fixed);
  for (auto& [time_checker_name, time_list] : time_list_of_functions_) {
    SortTimeList(&time_list);
    ss << time_checker_name << "\n";
    ss << std::setw(18) << std::setfill(' ') << "calls: " << time_list.size() << "\n";
    if (time_list.empty()) continue;
    ss << std::setw(18) << std::setfill(' ');
    ss << "min: ";
    ss << std::setw(15) << std::setfill(' ') << GetMin(time_list) << " [ms]\n";
    ss << std::setw(18) << std::setfill(' ');
    ss << "max: ";
    ss << std::setw(15) << std::setfill(' ') << GetMax(time_list) << " [ms]\n";
    ss << std::setw(18) << std::setfill(' ');
    ss << "avg: ";
    ss << std::setw(15) << std::setfill(' ') << GetAverage(time_list) << " [ms]\n";
    ss << std::setw(18) << std::setfill(' ');
    ss << "std: ";
    ss << std::setw(15) << std::setfill(' ') << GetStd(time_list) << " [ms]\n";
    ss << std::setw(18) << std::setfill(' ');
    ss << "total occupancy: ";
    ss << std::setw(15) << std::setfill(' ') << GetSum(time_list) << " [ms]\n";
    ss << " -----------------------------------------------\n";
  }
  std::cerr << ss.str();

  if (save_file_name_.size() > 0) {
    std::cerr << "Save file!\n";
  }
}

void TimeCheckerManager::SortTimeList(std::vector<double>* time_list) {
  std::sort(time_list->begin(), time_list->end());
}

double TimeCheckerManager::GetMin(const std::vector<double>& time_list) { return time_list.front(); }

double TimeCheckerManager::GetMax(const std::vector<double>& time_list) { return time_list.back(); }

double TimeCheckerManager::GetSum(const std::vector<double>& time_list) {
  double time_sum = 0.0;
  for (const auto& time : time_list) time_sum += time;
  return time_sum;
}

double TimeCheckerManager::GetAverage(const std::vector<double>& time_list) {
  if (time_list.empty()) return 0.0;
  return (GetSum(time_list) / static_cast<double>(time_list.size()));
}

double TimeCheckerManager::GetStd(const std::vector<double>& time_list) {
  if (time_list.size() <= 1) return 0.0;
  double squared_time_sum = 0.0;
  for (const auto& time : time_list) squared_time_sum += time * time;
  const double avg_time = GetAverage(time_list);
  return std::sqrt(squared_time_sum / static_cast<double>(time_list.size()) - avg_time * avg_time);
}

}  // namespace nonlinear_optimizer