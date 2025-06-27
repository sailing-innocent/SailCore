/**
 * @file main.cpp
 * @brief gltf reader
 * @author sailing-innocent
 * @date 2025-01-07
 */

// #include "SailScene/mesh.h"
// #include "SailScene/loader.h"

#include <argparse.hpp>
#include <filesystem>
#include <tiny_gltf.h>
#include <string>

using GLTFAsset = tinygltf::Model;
using GLTFLoader = tinygltf::TinyGLTF;

int main(int argc, char** argv) {
	// sail::gl::GUIRaster app;

	// parse
	argparse::ArgumentParser program("03.gltf_reader");
	// --asset the asset path
	program.add_argument("--asset")
		.help("the asset path")
		.default_value(std::string("../../data/assets"));

	program.add_argument("--file")
		.help("the gltf file")
		.default_value(std::string("models/Cube/Cube.gltf"))
		.required();

	try {
		program.parse_args(argc, argv);
	} catch (const std::runtime_error& err) {
		std::cout << err.what() << std::endl;
		std::cout << program;
		exit(0);
	}

	std::string file_path = program.get<std::string>("--file");
	auto gltf_path = std::filesystem::path(program.get<std::string>("--asset")) / file_path;

	GLTFAsset model;
	GLTFLoader loader;
	std::string err, warn;
	if (!loader.LoadASCIIFromFile(&model, &err, &warn, gltf_path.string())) {
		throw std::runtime_error(warn + err);
	}

	// print model info
	printf("model info:\n");
	printf("scene size: %lld\n", model.scenes.size());
	printf("scene nodes size: %lld\n", model.nodes.size());
	printf("scene meshes size: %lld\n", model.meshes.size());

	for (auto& mesh : model.meshes) {
		printf("mesh name: %s\n", mesh.name.c_str());
		printf("mesh primitives size: %lld\n", mesh.primitives.size());
		// vertex & index
		for (auto& primitive : mesh.primitives) {
			printf("primitive attributes size: %lld\n", primitive.attributes.size());

			printf("indices: %d\n", primitive.indices);
			auto index_accessor = model.accessors[primitive.indices];
			printf("index count: %lld\n", index_accessor.count);
			// access the data
			auto& buffer_view = model.bufferViews[index_accessor.bufferView];
			auto& buffer = model.buffers[buffer_view.buffer];
			auto* data = buffer.data.data() + buffer_view.byteOffset + index_accessor.byteOffset;
			// show the start and end of the data
			printf("index data: %d %d\n", *(uint16_t*)data, *(uint16_t*)(data + (index_accessor.count - 1) * index_accessor.ByteStride(buffer_view)));
			for (auto& attr : primitive.attributes) {
				printf("attribute name: %s\n", attr.first.c_str());
				auto accessor = model.accessors[primitive.attributes[attr.first]];
				printf("accessor count: %lld\n", accessor.count);
				printf("accessor type: %d\n", accessor.type);
				printf("accessor component type: %d\n", accessor.componentType);
				printf("accessor byte offset: %lld\n", accessor.byteOffset);
				printf("accessor byte stride: %d\n", accessor.ByteStride(model.bufferViews[accessor.bufferView]));
				printf("accessor buffer view: %d\n", accessor.bufferView);
				printf("accessor buffer view offset: %lld\n", model.bufferViews[accessor.bufferView].byteOffset);

				auto& buffer_view = model.bufferViews[accessor.bufferView];
				auto& buffer = model.buffers[buffer_view.buffer];
				auto* data = buffer.data.data() + buffer_view.byteOffset + accessor.byteOffset;
				// show the start and end of the data
				printf("data: %f %f\n", *(float*)data, *(float*)(data + (accessor.count - 1) * accessor.ByteStride(buffer_view)));
			}
		}
	}

	printf("scene materials size: %lld\n", model.materials.size());
	for (auto& mat : model.materials) {
		printf("material name: %s\n", mat.name.c_str());
		printf("material pbrMetallicRoughness baseColorFactor: %f %f %f %f\n", mat.pbrMetallicRoughness.baseColorFactor[0], mat.pbrMetallicRoughness.baseColorFactor[1], mat.pbrMetallicRoughness.baseColorFactor[2], mat.pbrMetallicRoughness.baseColorFactor[3]);
		// base color texture
		printf("material pbrMetallicRoughness baseColorTexture index: %d\n", mat.pbrMetallicRoughness.baseColorTexture.index);
		auto base_color_texture = model.textures[mat.pbrMetallicRoughness.baseColorTexture.index];
		printf("material pbrMetallicRoughness baseColorTexture sampler: %d\n", base_color_texture.sampler);
		auto base_color_image = model.images[base_color_texture.source];
		printf("material pbrMetallicRoughness baseColorTexture image: %s\n", base_color_image.uri.c_str());
		// show width and height
		printf("material pbrMetallicRoughness baseColorTexture image width: %d\n", base_color_image.width);
		printf("material pbrMetallicRoughness baseColorTexture image height: %d\n", base_color_image.height);
		// image data size
		printf("material pbrMetallicRoughness baseColorTexture image data size: %lld\n", base_color_image.image.size());

		printf("material pbrMetallicRoughness metallicFactor: %f\n", mat.pbrMetallicRoughness.metallicFactor);

		// metallic roughness texture
		printf("material pbrMetallicRoughness metallicRoughnessTexture index: %d\n", mat.pbrMetallicRoughness.metallicRoughnessTexture.index);
		auto metallic_roughness_texture = model.textures[mat.pbrMetallicRoughness.metallicRoughnessTexture.index];
		printf("material pbrMetallicRoughness metallicRoughnessTexture sampler: %d\n", metallic_roughness_texture.sampler);
		auto metallic_roughness_image = model.images[metallic_roughness_texture.source];
		printf("material pbrMetallicRoughness metallicRoughnessTexture image: %s\n", metallic_roughness_image.uri.c_str());
		// show width and height
		printf("material pbrMetallicRoughness metallicRoughnessTexture image width: %d\n", metallic_roughness_image.width);
		printf("material pbrMetallicRoughness metallicRoughnessTexture image height: %d\n", metallic_roughness_image.height);
		// image data size
		printf("material pbrMetallicRoughness metallicRoughnessTexture image data size: %lld\n", metallic_roughness_image.image.size());

		printf("material pbrMetallicRoughness roughnessFactor: %f\n", mat.pbrMetallicRoughness.roughnessFactor);
		printf("material normalTexture index: %d\n", mat.normalTexture.index);
		printf("material normalTexture scale: %f\n", mat.normalTexture.scale);
		printf("material occlusionTexture index: %d\n", mat.occlusionTexture.index);
		printf("material occlusionTexture strength: %f\n", mat.occlusionTexture.strength);
		printf("material emissiveFactor: %f %f %f\n", mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
		printf("material emissiveTexture index: %d\n", mat.emissiveTexture.index);
	}
	printf("scene textures size: %lld\n", model.textures.size());
	printf("scene images size: %lld\n", model.images.size());
	printf("scene animations size: %lld\n", model.animations.size());
	printf("scene skins size: %lld\n", model.skins.size());
	printf("scene cameras size: %lld\n", model.cameras.size());
	printf("scene lights size: %lld\n", model.lights.size());
	return 0;
}