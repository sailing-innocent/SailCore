-- Some C++ Special Case for Performance
function cpp_demo(name, deps)
    if deps == nil then
        deps = {}
    end
    target("cpp_demo_" .. name)
        set_kind("binary")
        add_files(name .. ".cpp")
        add_deps(deps)
    if is_plat("windows") then
        add_defines("WIN32_LEAN_AND_MEAN")
    end
end

cpp_demo("simd_sample")