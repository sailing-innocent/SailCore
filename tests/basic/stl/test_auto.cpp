/**
 * @file test_auto.cpp
 * @brief The C++11 auto test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"
#include <type_traits>

namespace sail::test {

struct Dummy {
	int var;
};

decltype(auto) func_0(Dummy& d) {
	return d.var;
}

decltype(auto) func_1(Dummy& d) {
	return (d.var);
}

int test_auto() {
	Dummy d{0};
	CHECK(std::is_same_v<decltype(func_0(d)), int>);
	CHECK(std::is_same_v<decltype(func_1(d)), int&>);
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic-stl") {
	TEST_CASE("cpp11-auto") {
		REQUIRE(sail::test::test_auto() == 0);
	}
}
