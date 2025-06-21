#pragma once
/**
  * @file array_ops.cuh
  * @brief The Array Operations Kernel
  * @author sailing-innocent
  * @date 2025-01-20
  */

#include <cuda_runtime_api.h>
#include <cutlass/array.h>

namespace sail::cut::kernel {

template<typename T, int N>
__global__ void test_array_clear(cutlass::Array<T, N>* ptr) {

	cutlass::Array<T, N> storage;

	storage.clear();

	ptr[threadIdx.x] = storage;
}

}// namespace sail::cut::kernel