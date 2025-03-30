#ifndef NONLINEAR_OPTIMIZER_TIME_CHECKER_H_
#define NONLINEAR_OPTIMIZER_TIME_CHECKER_H_

#include <chrono>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace nonlinear_optimizer {

#define CHECK_EXEC_TIME_FROM_HERE \
  nonlinear_optimizer::TimeChecker time_checker(__FILE__, __func__);

class TimeChecker {
 public:
  TimeChecker(const std::string& file_name, const std::string& function_name);
  ~TimeChecker();

 private:
  const std::string name_;
  const std::chrono::high_resolution_clock::time_point start_time_;
};

class TimeCheckerManager {
  friend class TimeChecker;

 public:
  TimeCheckerManager(const TimeCheckerManager& time_checker_manager) = delete;

  static void SetMinTimeThreshold(const double min_time_threshold);
  static void SaveFile(const std::string& file_name);
  static TimeCheckerManager* GetSingletonInstance();

 protected:
  void RegisterTime(const std::string& time_checker_name, const double time);

 private:
  TimeCheckerManager();
  ~TimeCheckerManager();

  void SortTimeList(std::vector<double>* time_list);
  double GetMin(const std::vector<double>& time_list);
  double GetMax(const std::vector<double>& time_list);
  double GetSum(const std::vector<double>& time_list);
  double GetAverage(const std::vector<double>& time_list);
  double GetStd(const std::vector<double>& time_list);

  std::map<std::string, std::vector<double>> time_list_of_functions_;

  static double min_time_threshold_;
  static std::string save_file_name_;
};

}  // namespace nonlinear_optimizer

// #define log(fmt, ...)                                                 \
//   printf("[%s: %d][%s] " fmt "\t\t\t (%s, %s)\n", __FILE__, __LINE__, \
//          __func__, __DATE__, __TIME__);

#endif  // NONLINEAR_OPTIMIZER_TIME_CHECKER_H_
