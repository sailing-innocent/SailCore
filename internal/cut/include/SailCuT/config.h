#pragma once
#include "SailCu/config.h"

namespace sail::cu {

// cublas API error checking
#define CUBLAS_CHECK(err)                                                        \
	do {                                                                         \
		cublasStatus_t err_ = (err);                                             \
		if (err_ != CUBLAS_STATUS_SUCCESS) {                                     \
			std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
			throw std::runtime_error("cublas error");                            \
		}                                                                        \
	} while (0)

}// namespace sail::cu
