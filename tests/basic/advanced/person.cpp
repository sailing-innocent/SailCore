#include "person.hpp"

namespace sail::test {

struct Person::Impl {
	int id = 1;
};

Person::Person() {
	mp_impl = std::make_unique<Impl>();
}
Person::~Person() {}
int Person::id() const {
	return mp_impl->id;
}

}// namespace sail::test