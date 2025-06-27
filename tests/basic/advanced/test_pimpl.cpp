/**
 * @file test_pimpl.cpp
 * @brief The Pimpl test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"
#include "person.hpp"

namespace sail::test {

int test_pimpl() {
	Person person;
	CHECK(person.id() == 1);
	return 0;
}

}// namespace sail::test

TEST_SUITE("basic::design") {
	TEST_CASE("pimpl") {
		REQUIRE(sail::test::test_pimpl() == 0);
	}
}