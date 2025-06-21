/**
 * @file test_matrix.cu
 * @brief Test the matrix operations in cuBLAS
 * @author sailing-innocent
 * @date 2025-02-14
 */

#include "test_util.h"
#include <vector>
#include <cublas_v2.h>
#include "SailCuT/dummy.h"

namespace sail::test {

using namespace sail::cu;

bool test_gemm() {
	cublasHandle_t cublasH = nullptr;
	cudaStream_t stream = nullptr;
	// A = | 1.0 2.0 |
	//     | 3.0 4.0 |
	const std::vector<float> A = {1.0, 2.0, 3.0, 4.0};
	// B = | 1.0 2.0 |
	//     | 3.0 4.0 |
	const std::vector<float> B = {1.0, 2.0, 3.0, 4.0};
	// C = | 0.0 0.0 |
	//     | 0.0 0.0 |
	const std::vector<float> C = {0.0, 0.0, 0.0, 0.0};
	const int m = 2;
	const int n = 2;
	const int k = 2;
	const float alpha = 1.0;
	const float beta = 0.0;
	// step 1: create cublas handle, bind a stream
	CUBLAS_CHECK(cublasCreate(&cublasH));
	CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
	CUBLAS_CHECK(cublasSetStream(cublasH, stream));

	// step 2: copy data to device
	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_C = nullptr;
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(float) * A.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), sizeof(float) * B.size()));
	CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_C), sizeof(float) * C.size()));
	CUDA_CHECK(cudaMemcpyAsync(d_A, A.data(), sizeof(float) * A.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_B, B.data(), sizeof(float) * B.size(), cudaMemcpyHostToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(d_C, C.data(), sizeof(float) * C.size(), cudaMemcpyHostToDevice, stream));

	// step 3: compute
	CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m));

	// step 4: copy result back to host
	std::vector<float> result(C.size());
	CUDA_CHECK(cudaMemcpyAsync(result.data(), d_C, sizeof(float) * C.size(), cudaMemcpyDeviceToHost, stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));
	// result = | 7.0 10.0 |
	//          | 15.0 22.0 |
	CHECK(result[0] == 7.0);
	CHECK(result[1] == 10.0);
	CHECK(result[2] == 15.0);
	CHECK(result[3] == 22.0);

	// free resources
	CUDA_CHECK(cudaFree(d_A));
	CUDA_CHECK(cudaFree(d_B));
	CUDA_CHECK(cudaFree(d_C));
	CUBLAS_CHECK(cublasDestroy(cublasH));

	return true;
}

}// namespace sail::test

TEST_SUITE("cublas") {
	TEST_CASE("matrix_ops") {
		CHECK(sail::test::test_gemm());
	}
}