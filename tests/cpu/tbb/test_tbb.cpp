/**
 * @file test_tbb.cpp
 * @brief The Test TBB Basic
 * @author sailing-innocent
 * @date 2025-02-10
 */

#include "test_util.h"
#include "SailCPU/reduce.h"
#include <numeric>

namespace sail::test {

bool test_tbb() {
	CHECK(cpu::reduce_sum(0, 100) == 5050 - 100);
	return true;
}

}// namespace sail::test

TEST_SUITE("cpu") {
	TEST_CASE("basic_tbb") {
		CHECK(sail::test::test_tbb());
	}
}