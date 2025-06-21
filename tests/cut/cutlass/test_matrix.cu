/**
  * @file test_matrix.cu
  * @brief The Matrix Operations Test
  * @author sailing-innocent
  * @date 2025-01-20
  */
#include "test_util.h"
#include "cutlass/matrix.h"

namespace sail::test {

bool test_cutlass_matrix() {
	cutlass::Matrix<float, 4, 4> matrix;
	return true;
}

}// namespace sail::test

TEST_SUITE("cutlass") {
	TEST_CASE("basic_matrix") {
		CHECK(sail::test::test_cutlass_matrix());
	}
}