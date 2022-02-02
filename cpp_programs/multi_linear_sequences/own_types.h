#pragma once

template<typename InputType, typename ReturnType>
struct ThreadDataGeneric;

typedef struct InputTypeOwn_ {
  U32 dim;
  U32 n;
  U32 m;
  U64 k_idx_start;
  U64 k_idx_end;
} InputTypeOwn;

typedef struct ReturnTypeOwn_ {
  map<U32, U32> map_len_cycle_to_count;
} ReturnTypeOwn;

typedef struct InputTypeOwn_ InputTypeOwn;
typedef struct ReturnTypeOwn_ ReturnTypeOwn;
