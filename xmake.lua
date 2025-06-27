set_xmakever("2.9.2")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
engine_version = "0.1.0"
set_languages("c++20")

includes("xmake/default_options.lua")
includes("xmake/rules.lua")

if is_mode("debug") then
    set_targetdir("bin/debug")
    set_runtimes("MDd")
elseif is_mode("releasedbg") then
    set_targetdir("bin/releasedbg")
    set_runtimes("MD")
else
    set_targetdir("bin/release")
    set_runtimes("MD")
end


add_requires("glfw", {configs = {vulkan = true}})
add_requires("tbb")
add_requires("embree")
add_requires("yyjson")
add_requires("glm") 
add_requires("eastl")
add_requires("imgui") 
add_requires("pybind11")
add_requires("libhv")
add_requires("spdlog")

--- assets
includes("assets")

-------------------------------------
--- CORE MODULES
-------------------------------------

includes("external")
includes("modules")
includes("internal")
-------------------------------------
--- TESTS AND EXAMPLES
-------------------------------------
includes("tests")
includes("examples")
