/**
 * @file test_operator.cpp
 * @brief The Operator Overloading test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"

namespace sail::test {

class point {
private:
	int m_x, m_y;

public:
	point(int a = 0, int b = 0);
	friend point operator-(point p);
	int& x() { return m_x; }
	int& y() { return m_y; }
};
point::point(int a, int b) {
	m_x = a;
	m_y = b;
}
point operator-(point p) {
	point temp;
	temp.x() = -p.x();
	temp.y() = -p.y();
	return temp;
}
}// namespace sail::test

TEST_SUITE("basic-dummy") {
	TEST_CASE("operator") {
		sail::test::point p1(10, 15);
		sail::test::point p2 = -p1;
		REQUIRE(p2.x() == -10);
		REQUIRE(p2.y() == -15);
	}
}
