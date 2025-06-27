/**
 * @file 03.simple_threads.cpp
 * @brief A Simple Multi-Thread Demo
 * @author sailing-innocent
 * @date 2025-02-09
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <vector>
#include <random>

std::mutex values_mtx;
std::mutex cout_mtx;
std::vector<int> values;

int randGen(const int& min, const int& max) {
	static thread_local std::mt19937 generator(std::hash<std::thread::id>()(std::this_thread::get_id()));
	std::uniform_int_distribution<int> distribution(min, max);
	return distribution(generator);
}

void threadFunc(int tid) {
	cout_mtx.lock();
	std::cout << "thread " << tid << " started" << std::endl;
	cout_mtx.unlock();

	values_mtx.lock();
	int val = values[0];
	values_mtx.unlock();

	int rval = randGen(0, 10);
	val += rval;
	cout_mtx.lock();
	std::cout << "thread " << tid << " added " << rval << "; New Values: " << val << "." << std::endl;
	cout_mtx.unlock();

	values_mtx.lock();
	values.push_back(val);
	values_mtx.unlock();
}

int main() {
	values.push_back(42);
	std::thread tr1(threadFunc, 1);
	std::thread tr2(threadFunc, 2);
	std::thread tr3(threadFunc, 3);
	std::thread tr4(threadFunc, 4);
	tr1.join();
	tr2.join();
	tr3.join();
	tr4.join();
}