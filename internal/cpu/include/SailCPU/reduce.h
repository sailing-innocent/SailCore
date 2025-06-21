#pragma once
/**
 * @file reduce.h
 * @brief The Basic Reduce Operation
 * @author sailing-innocent
 * @date 2025-02-10
 */

#include "SailCPU/config.h"

namespace sail::cpu {

int SAIL_CPU_API reduce_sum(int from_n, int to_n) noexcept;

}// namespace sail::cpu