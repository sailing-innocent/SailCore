/**
  * @file vertex_add.cu
  * @brief The Basic Hello World Example to Learn NSight Profiler
  * @author sailing-innocent
  * @date 2025-01-18
  */

#include <iostream>
#include <vector>

#define N 10000

__global__ void add(int* a, int* b, int* c) {
	int tid = blockIdx.x;
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}

int main() {
	std::vector<int> a(N, 1);
	std::vector<int> b(N, 2);
	std::vector<int> c(N, 0);

	int* d_a;
	int* d_b;
	int* d_c;

	cudaMalloc(&d_a, N * sizeof(int));
	cudaMalloc(&d_b, N * sizeof(int));
	cudaMalloc(&d_c, N * sizeof(int));

	cudaMemcpy(d_a, a.data(), N * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b.data(), N * sizeof(int), cudaMemcpyHostToDevice);

	add<<<N, 1>>>(d_a, d_b, d_c);

	cudaMemcpy(c.data(), d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}