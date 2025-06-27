/**
 * @file test_tuple.cpp
 * @brief The C++11 tuple test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"

#include <tuple>

TEST_SUITE("basic::semantic") {
	TEST_CASE("cpp11::tuple") {
		std::tuple<int, int> a{1, 2};
		REQUIRE(std::get<0>(a) == 1);
		REQUIRE(std::get<1>(a) == 2);
	}
}
