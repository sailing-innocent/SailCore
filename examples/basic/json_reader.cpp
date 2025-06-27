/**
 * @file json_reader.cpp
 * @brief The json reader based on yyjson
 * @author sailing-innocent
 * @date 2025-03-04
 */

#include "yyjson.h"
#include <argparse.hpp>
#include <iostream>

int main(int argc, char** argv) {

	argparse::ArgumentParser program("03.json_reader");
	program.add_argument("--file")
		.help("the json file")
		.default_value(std::string("../../data/assets/samples/sample.json"))
		.required();

	try {
		program.parse_args(argc, argv);
	} catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string file_path = program.get<std::string>("--file");
	FILE* file = fopen(file_path.c_str(), "rb");
	if (!file) {
		std::cerr << "failed to open file: " << file_path << std::endl;
		return -1;
	}

	fseek(file, 0, SEEK_END);
	size_t file_size = ftell(file);
	fseek(file, 0, SEEK_SET);

	char* buffer = (char*)malloc(file_size);

	if (!buffer) {
		std::cerr << "failed to allocate memory" << std::endl;
		fclose(file);
		return -1;
	}

	if (fread(buffer, 1, file_size, file) != file_size) {
		std::cerr << "failed to read file" << std::endl;
		fclose(file);
		free(buffer);
		return -1;
	}

	fclose(file);

	yyjson_doc* doc = yyjson_read(buffer, file_size, 0);
	if (!doc) {
		std::cerr << "failed to parse json" << std::endl;
		free(buffer);
		return -1;
	}

	yyjson_val* root = yyjson_doc_get_root(doc);
	if (!root) {
		std::cerr << "failed to get root" << std::endl;
		yyjson_doc_free(doc);
		free(buffer);
		return -1;
	}

	if (yyjson_get_type(root) != YYJSON_TYPE_OBJ) {
		std::cerr << "root is not an object" << std::endl;
		yyjson_doc_free(doc);
		free(buffer);
		return -1;
	}

	return 0;
}