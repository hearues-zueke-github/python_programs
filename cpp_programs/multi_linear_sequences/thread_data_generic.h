#pragma once

#include <assert.h>
#include <condition_variable>

#include "utils.h"
#include "own_types.h"
#include "multi_linear_sequences.h"

typedef enum ThreadState_ {
  NONE = 0,
  START,
  END,
} ThreadState;

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
  ThreadDataGeneric(std::mutex& mutex_, std::condition_variable& cond_var_, void (*function_)(InputType&, ReturnType&));
  void start();
};

template struct ThreadDataGeneric<InputTypeOwn, ReturnTypeOwn>;
