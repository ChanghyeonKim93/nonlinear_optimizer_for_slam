#ifndef NONLINEAR_OPTIMIZER_MULTI_THREAD_EXECUTOR_H_
#define NONLINEAR_OPTIMIZER_MULTI_THREAD_EXECUTOR_H_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace std ::chrono_literals;

#define MULTI_THREAD_EXECUTOR_VERBOSE false

#ifndef PrintInfo
inline void PrintInfoImpl(const std::string& str, const std::string& func_str) {
  std::stringstream ss;
  ss << "[ThreadId:" << std::this_thread::get_id() << "][INFO] \"" << func_str
     << "()\": " << str << "\n";
  std::cerr << ss.str();
}
#define PrintInfo(str) PrintInfoImpl(str, std::string(__func__));
#endif

#ifndef ThrowError
inline void ThrowErrorImpl(const std::string& str,
                           const std::string& func_str) {
  std::stringstream ss;
  ss << "[ThreadId:" << std::this_thread::get_id() << "][Error] \"" << func_str
     << "()\": " << str << "\n";
  throw std::runtime_error(ss.str());
}
#define ThrowError(str) ThrowErrorImpl(str, std::string(__func__));
#endif

class MultiThreadExecutor {
 public:
  enum class Status { kRun = 0, kTerminate = 1 };

 public:
  MultiThreadExecutor() = delete;

  explicit MultiThreadExecutor(const int num_threads) : status_(Status::kRun) {
    worker_threads_.reserve(num_threads);
    for (int index = 0; index < num_threads; ++index)
      worker_threads_.emplace_back(
          [this, index]() { ActivateWorkerThread(index); });
  }

  explicit MultiThreadExecutor(
      const int num_threads,
      const std::vector<int>& processor_numbers_for_each_thread)
      : status_(Status::kRun) {
    if (num_threads !=
        static_cast<int>(processor_numbers_for_each_thread.size()))
      ThrowError("processor_numbers_for_each_thread.size() != num_threads");

    worker_threads_.reserve(num_threads);
    for (int index = 0; index < num_threads; ++index) {
      worker_threads_.emplace_back(
          [this, index]() { ActivateWorkerThread(index); });
      AllocateProcessorsForEachThread(worker_threads_[index],
                                      processor_numbers_for_each_thread[index]);
    }
  }

  ~MultiThreadExecutor() {
    status_ = Status::kTerminate;
    WakeUpAllThreads();
    for (auto& worker_thread : worker_threads_) {
      worker_thread.join();
      PrintInfo("Executor thread successfully joins.");
    }
  }

  template <typename FunctionName, typename... Arguments>
  std::future<typename std::invoke_result<FunctionName, Arguments...>::type>
  Execute(FunctionName&& func, Arguments&&... args) {
    using ReturnType =
        typename std::invoke_result<FunctionName, Arguments...>::type;
    auto task_ptr = std::make_shared<std::packaged_task<ReturnType()>>(
        std::bind(std::forward<FunctionName>(func),
                  std::forward<Arguments>(args)...));
    std::future<ReturnType> future_result = task_ptr->get_future();

    mutex_for_cv_.lock();
    task_queue_.emplace_back([task_ptr]() { (*task_ptr)(); });
    mutex_for_cv_.unlock();

    WakeUpOneThread();
    return future_result;
  }

  void PurgeAllTasks() {
    std::unique_lock<std::mutex> local_lock(mutex_for_cv_);
    task_queue_.clear();
  }

  int GetNumOfOngoingTasks() { return num_of_ongoing_tasks_; }

  int GetNumOfQueuedTasks() {
    std::unique_lock<std::mutex> local_lock(mutex_for_cv_);
    return static_cast<int>(task_queue_.size());
  }

  int GetNumOfTotalThreads() {
    std::unique_lock<std::mutex> local_lock(mutex_for_cv_);
    return static_cast<int>(worker_threads_.size());
  }

  int GetNumOfAwaitingThreads() {
    return (static_cast<int>(worker_threads_.size()) - num_of_ongoing_tasks_);
  }

 private:
  void WakeUpOneThread() { cv_.notify_one(); }
  void WakeUpAllThreads() { cv_.notify_all(); }
  bool AllocateProcessorsForEachThread(std::thread& input_thread,
                                       const int logical_core_index) {
    const int num_logical_cores =
        static_cast<int>(std::thread::hardware_concurrency());
    PrintInfo("# of logical cores: " + std::to_string(num_logical_cores));
    if (logical_core_index >= num_logical_cores) {
      PrintInfo("Exceed the maximum number of logical processors!");
      return false;
    }

    cpu_set_t cpu_set;
    CPU_ZERO(&cpu_set);
    CPU_SET(logical_core_index, &cpu_set);
    int result = pthread_setaffinity_np(input_thread.native_handle(),
                                        sizeof(cpu_set_t), &cpu_set);
    if (result != 0) {
      PrintInfo("Fail to allocate processor. pthread_setaffinity_np yields: " +
                std::to_string(result));
      return false;
    }
    return true;
  }
  void ActivateWorkerThread(const int thread_index) {
    while (true) {
      std::unique_lock<std::mutex> local_lock(mutex_for_cv_);
      cv_.wait(local_lock, [this]() {
        return (!task_queue_.empty() || (status_ == Status::kTerminate));
      });

      if (status_ == Status::kTerminate) break;

      std::function<void()> new_task = std::move(task_queue_.front());
      task_queue_.pop_front();
      local_lock.unlock();
      ++num_of_ongoing_tasks_;

      if (MULTI_THREAD_EXECUTOR_VERBOSE)
        PrintInfo("Thread [" + std::to_string(thread_index) + "] works.");
      new_task();  // Do the job!
    }
    PrintInfo("While loop of executor thread is terminated.");
  }

 private:
  std::atomic<Status> status_;

  std::mutex mutex_for_cv_;
  std::condition_variable cv_;

  std::vector<std::thread> worker_threads_;
  std::deque<std::function<void()>> task_queue_;

  std::atomic<int> num_of_ongoing_tasks_{0};
};

#endif  // NONLINEAR_OPTIMIZER_MULTI_THREAD_EXECUTOR_H_