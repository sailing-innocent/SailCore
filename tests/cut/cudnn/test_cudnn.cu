/**
  * @file test_cudnn.cpp
  * @brief test the cudnn library
  * @author sailing-innocent
  * @date 2025-01-18
  */

#include "test_util.h"
#include <cudnn.h>

namespace sail::test {

bool test_basic_cudnn() {
	auto v = cudnnGetMaxDeviceVersion();
	CHECK(v == 900);
	return true;
}

}// namespace sail::test

TEST_SUITE("cudnn") {
	TEST_CASE("basic_cudnn") {
		CHECK(sail::test::test_basic_cudnn());
	}
}