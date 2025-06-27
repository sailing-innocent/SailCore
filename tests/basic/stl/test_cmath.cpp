/**
 * @file test_cmath.cpp
 * @brief The C++11 cmath test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"
#include <cmath>
#include <numbers>

TEST_SUITE("basic-stl") {
	TEST_CASE("cmath") {
		float PI = std::numbers::pi_v<float>;
		REQUIRE(abs(tanf(45.0f / 180.0f * PI) - 1.0f) < 0.001f);
		REQUIRE(abs(tanf(30.0f / 180.0f * PI) - 0.57735026919f) < 0.001f);
	}
}
