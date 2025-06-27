includes("cpp")


target("multi_process")
    set_kind("binary")
    add_packages("libhv")
    add_files("multi_process.cpp")
target_end()


target("use_cpp23module")
    set_kind("binary")
    add_files("use_cpp23module.cpp")
    set_languages("cxx23")
    set_toolchains("msvc")
    set_policy("build.c++.modules", true)
target_end()

includes("rtow") -- ray tracing one weekend

target("gltf_reader")
    set_kind("binary")
    add_files("gltf_reader.cpp")
    add_deps("external_tiny_gltf_util")
    add_deps("external_argparse")
target_end()

target("gltf_writer")
    set_kind("binary")
    add_files("gltf_writer.cpp")
    add_deps("external_tiny_gltf_util")
target_end()

target("json_reader")
    set_kind("binary")
    add_files("json_reader.cpp")
    add_packages("yyjson")
    add_deps("external_argparse")
target_end()

target("ply_reader")
    set_kind("binary")
    add_files("ply_reader.cpp")
    add_deps("external_happly")
    add_deps("external_argparse")
target_end()

target("ray_marcher")
    set_kind("binary")
    add_files("ray_marcher.cpp")
target_end()