export module ThreadDataGeneric;

#include <assert.h>
#include <condition_variable>

#include "utils.h"

export
typedef enum ThreadState_ {
  NONE = 0,
  START,
  END,
} ThreadState;

export
template<typename InputType, typename ReturnType>
struct ThreadDataGeneric {
  std::mutex& mutex;
  std::condition_variable& cond_var;
  bool main_notify_thread;
  bool thread_notify_main;
  ThreadState state;
  InputType var_input;
  ReturnType var_return;
  void (*function)(InputType& var_input, ReturnType& var_return);
  ThreadDataGeneric(
    std::mutex& mutex_, std::condition_variable& cond_var_,
    void (*function_)(InputType&, ReturnType&)
  ) : mutex(mutex_), cond_var(cond_var_),
      main_notify_thread(false), thread_notify_main(false),
      state(NONE),
      var_input(), var_return(), // {
      function(function_) {
  }
  void start() {
    while (true) {
      { // wait in thread on start of main thread
        std::unique_lock<std::mutex> lk(this->mutex);
        this->cond_var.wait(lk, [&]{ return this->main_notify_thread; });
      }

      bool is_end = false;  
      switch (this->state) {
        case ThreadState::NONE:
          assert(false && "ThreadState state should not be NONE!");
          break;
        case ThreadState::START:
          // execute the function in the thread!
          // calcCycleLengthAmounts(this->var_input, this->var_return);
          (*this->function)(this->var_input, this->var_return);
          break;
        case ThreadState::END:
          is_end = true;
          break;
      }

      { // notify the main thread that this thread worker is finished
        std::lock_guard<std::mutex> lk(this->mutex);
        this->thread_notify_main = true;
        this->main_notify_thread = false;
        this->state = ThreadState::NONE;
      }
      this->cond_var.notify_one();

      if (is_end) {
        return;
      }
    }
  }
};
