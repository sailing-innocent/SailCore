#pragma once
#include "SailBase/config.h"

#define SAIL_HOST_DEVICE __host__ __device__

// CUDA API error checking
#define CUDA_CHECK(err)                                                        \
	do {                                                                       \
		cudaError_t err_ = (err);                                              \
		if (err_ != cudaSuccess) {                                             \
			std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
			throw std::runtime_error("CUDA error");                            \
		}                                                                      \
	} while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                        \
	do {                                                                         \
		cublasStatus_t err_ = (err);                                             \
		if (err_ != CUBLAS_STATUS_SUCCESS) {                                     \
			std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
			throw std::runtime_error("cublas error");                            \
		}                                                                        \
	} while (0)