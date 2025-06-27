/**
 * @file test_concepts.cpp
 * @brief The C++20 Concepts test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"
#include <string>
#include <cstddef>
#include <concepts>

namespace sail::test {

template<typename T>
concept always_satisfied = true;

template<typename T>
concept Hashable = requires(T a) {
	{
		std::hash<T>{}(a)
	} -> std::convertible_to<std::size_t>;
};

struct meow {
};

template<Hashable T>
bool f(T t) {
	return true;
}

int test_hashable_success() {
	using namespace std::literals;
	using std::operator""s;
	try {
		CHECK(f("abc"s));
	} catch (...) {
		return 1;
	}
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic-stl") {
	TEST_CASE("cpp20-concepts") {
		REQUIRE(sail::test::test_hashable_success() == 0);
	}
}
