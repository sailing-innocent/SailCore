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