/**
 * @file test_raii.cpp
 * @brief The C++11 RAII test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"

#include <memory>

namespace sail::test {

class Dummy {
public:
	int value = 1;
};

struct MyStruct {
	MyStruct() = default;
	MyStruct(int _idx)
		: index(_idx) {
	}
	int index = 0;
	int getIndex() { return index; }
};
}// namespace sail::test

TEST_SUITE("cpp11") {
	TEST_CASE("cpp11::unique_ptr") {
		std::unique_ptr<sail::test::Dummy> ptr;
		ptr = std::make_unique<sail::test::Dummy>(std::move(sail::test::Dummy{}));
		auto ptr_moved = std::move(ptr);
		CHECK(ptr_moved->value == 1);
	}
	TEST_CASE("cpp11::shared_ptr") {
		using namespace sail::test;
		MyStruct mystruct = MyStruct(1);
		REQUIRE(mystruct.getIndex() == 1);

		std::shared_ptr<MyStruct> p = std::make_shared<MyStruct>(mystruct);
		REQUIRE(p->getIndex() == 1);

		std::shared_ptr<MyStruct> np = p;
		REQUIRE(p->index == 1);
	}
}