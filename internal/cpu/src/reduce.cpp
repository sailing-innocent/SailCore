/**
 * @file reduce.cpp
 * @brief The Basic Reduce Implementation
 * @author sailing-innocent
 * @date 2025-02-10
 */

#include "SailCPU/reduce.h"
#include "tbb/tbb.h"

namespace sail::cpu {
int SAIL_CPU_API reduce_sum(int from_n, int to_n) noexcept {
	int sum = oneapi::tbb::parallel_reduce(
		oneapi::tbb::blocked_range<int>(from_n, to_n), 0,
		[](oneapi::tbb::blocked_range<int> const& r, int init) -> int {
		for (int v = r.begin(); v != r.end(); v++) {
			init += v;
		}
		return init;
	},
		[](int lhs, int rhs) -> int {
		return lhs + rhs;
	});
	return sum;
}

}// namespace sail::cpu