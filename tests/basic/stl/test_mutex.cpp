/**
 * @file test_mutex.cpp
 * @brief The C++11 mutex test file
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include "test_util.h"
#include <mutex>

namespace sail::test {

volatile int counter = 0;
std::mutex mtx;

void attempt_10k_increases() {
	for (int i = 0; i < 10000; ++i) {
		if (mtx.try_lock()) {
			++counter;
			mtx.unlock();
		}
	}
}

void atomic_10k_increases() {
	for (int i = 0; i < 10000; ++i) {
		std::lock_guard<std::mutex> lock(mtx);
		++counter;
	}
}

bool test_mutex() {
	std::thread threads[10];
	for (int i = 0; i < 10; ++i) {
		// threads[i] = std::thread(attempt_10k_increases);
		threads[i] = std::thread(atomic_10k_increases);
	}
	for (auto& th : threads) {
		th.join();
	}
	CHECK(counter == 100000);
	return true;
};

}// namespace sail::test

TEST_SUITE("stl::mutex") {
	TEST_CASE("cpp11::mutex") {
		CHECK(sail::test::test_mutex());
	}
}