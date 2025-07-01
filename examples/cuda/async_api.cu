#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

__global__ void increment_kernel(int* g_data, int inc_value) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	g_data[idx] += inc_value;
}

int main(int argc, char** argv) {
	int devID;
	cudaDeviceProp deviceProp;
}