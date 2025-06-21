/**
  * @file test_array.cu
  * @brief Test the array operations in cuBLAS
  * @author sailing-innocent
  * @date 2025-01-18
  */

#include "test_util.h"

#include <cuda_runtime_api.h>
#include <vector>
#include <cublas_v2.h>
#include "SailCuT/dummy.h"

namespace sail::test {

using namespace sail::cu;

template<CommonValueType T>
int cublasIMax(cublasHandle_t handle, int n, const T* x, int incx, int* result) {
	// switch (T)
	if constexpr (std::is_same_v<T, float>) {
		CUBLAS_CHECK(cublasIsamax(handle, n, x, incx, result));
	} else if constexpr (std::is_same_v<T, double>) {
		CUBLAS_CHECK(cublasIdamax(handle, n, x, incx, result));
	} else if constexpr (std::is_same_v<T, cuComplex>) {
		CUBLAS_CHECK(cublasIcamax(handle, n, x, incx, result));
	} else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
		CUBLAS_CHECK(cublasIzamax(handle, n, x, incx, result));
	} else {
		throw std::runtime_error("Unsupported type");
	}
	return 0;
}

template<CommonValueType T>
bool test_cublas_array() {
	// using namespace sail::cu;
	cublasHandle_t cublasH = nullptr;
	cudaStream_t stream = nullptr;
	// A = | 1.0 2.0 3.0 4.0 |
	const std::vector<T> A = {T(1), T(2), T(3), T(4)};
	const int incx = 1;
	int result = 0.0;
	// step 1: create cublas handle, bind a stream
	CUBLAS_CHECK(cublasCreate(&cublasH));
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUBLAS_CHECK(cublasSetStream(cublasH, stream));

	// step 2: copy data to device
	T* d_A = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * A.size()));
	CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(T) * A.size(), cudaMemcpyHostToDevice, stream));

	// step 3: compute
	cublasIMax<T>(cublasH, A.size(), d_A, incx, &result);// return the max value index to a host pointer

	CUDA_CHECK(cudaStreamSynchronize(stream));
	// result = 4
	CHECK(result == 4);

	// free resources
	CUDA_CHECK(cudaFree(d_A));
	CUBLAS_CHECK(cublasDestroy(cublasH));
	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaDeviceReset());
	return true;
}

}// namespace sail::test

TEST_SUITE("cublas") {
	TEST_CASE("basic_array") {
		CHECK(sail::test::test_cublas_array<double>());
		CHECK(sail::test::test_cublas_array<float>());
	}
}