/**
  * @file test_array.cu
  * @brief Cutlass Array Test
  * @author sailing-innocent
  * @date 2025-01-18
  */

#include "SailCuT/core/alloc.h"
#include "test_util.h"
// #include "cutlass/array.h"

namespace sail::test {

bool test_cutlass_array() {
	sail::cut::DeviceAllocation<cutlass::Array<float, 32>> output(static_cast<size_t>(32));
	CHECK(output.size() == 32);
	// // release
	output.release();
	CHECK(output.size() == 0);
	return true;
}

}// namespace sail::test

TEST_SUITE("cutlass") {
	TEST_CASE("basic_array") {
		CHECK(sail::test::test_cutlass_array());
	}
}