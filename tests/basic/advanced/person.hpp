#pragma once
#include <memory>

namespace sail::test {

class Person {
public:
	Person();
	~Person();
	// interface method
	[[nodiscard]] int id() const;

private:
	struct Impl;// hide details in impl
	std::unique_ptr<Impl> mp_impl;
};

}// namespace sail::test