/**
 * @file test_set.cpp
 * @brief The C++11 set test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"

#include <set>

TEST_SUITE("basic-stl") {
	TEST_CASE("set") {
		std::set<int> s = {1, 2};
		REQUIRE(s.size() == 2);
		s.insert(3);
		REQUIRE(s == std::set<int>{1, 2, 3});
	}
}
