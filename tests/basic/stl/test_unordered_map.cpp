/**
 * @file test_unordered_map.cpp
 * @brief The C++11 unordered_map test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"
#include <unordered_map>

TEST_SUITE("basic::containers") {
	TEST_CASE("cpp11::unordered_map") {
		std::unordered_map<int, int> m;
		m[1] = 2;
		m[2] = 3;
		REQUIRE(m[1] == 2);
		REQUIRE(m[2] == 3);
	}
}