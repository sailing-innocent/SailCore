/**
 * @file test_ply_io.cpp
 * @brief Test Suite for Happily Ply IO
 * @author sailing-innocent
 * @date 2024-11-13
 */

#include <string>
#include <argparse.hpp>
#include "happly.h"

int main(int argc, char** argv) {
	std::string ply_path = "../../data/assets/models/cube.ply";
	argparse::ArgumentParser program("ply_reader");
	program.add_argument("--ply")
		.help("the ply file path")
		.default_value(ply_path);
	try {
		program.parse_args(argc, argv);
	} catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}
	ply_path = program.get<std::string>("--ply");

	happly::PLYData plyIn(ply_path);
	std::vector<float> vx = plyIn.getElement("vertex").getProperty<float>("x");
	std::cout << "Number of vertices: " << vx.size() << std::endl;
	return 0;
}
