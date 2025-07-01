/**
 * @file thread_local.cpp
 * @brief The Thread Local Storage (TLS) example.
 * @author sailing-innocent
 * @date 2025-07-01
 */

#include <string>
#include <iostream>
#include <thread>

class A {
public:
	A() {}
	~A() {}
	void test(const std::string& name) {
		thread_local int count = 0;
		++count;
		// std::cout << "\nThread: " << name << ", Count: " << count << std::endl; // cout is not thread-safe, will cause unexpected behavior in multithreaded environment, uncomment this line to see the issue
		printf("\nThread: %s, Count: %d\n", name.c_str(), count);
	}
};

void func(const std::string& name) {
	A a;
	for (int i = 0; i < 5; ++i) {
		a.test(name);
	}
	A a2;
	a2.test(name);
}

int main() {
	std::thread t1(func, "T1");
	std::thread t2(func, "T2");

	t1.join();
	t2.join();

	std::thread t3(func, "T3");
	t3.join();

	return 0;
}