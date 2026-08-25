#pragma once

#ifdef LIB_DYNEIN_CELL_MODEL_DEBUG
#include <test_utils/test_utils.h>

// static test_utils::DebugRand<double> drand;
// #define prob_dist(rng) drand()

inline constexpr bool DYNEIN_CELL_MODEL_DEBUG_CPP = true;
#else
inline constexpr bool DYNEIN_CELL_MODEL_DEBUG_CPP = false;
#endif
