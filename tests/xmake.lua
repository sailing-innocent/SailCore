function sail_add_test(folder, name, deps, opts)
    deps = deps or {}
    opts = opts or {
        enable_cpp = true,
        enable_cu = false
    }
    target("test_" .. folder .. "_" .. name)
        set_kind("binary")
        set_group("02.tests/" .. folder .. "_" .. name)
        on_load(function (target)
            for k, v in pairs(opt) do
                target:set(k, v)
            end 
            target:set("exceptions", "cxx")
            for _, dep in ipairs(deps) do
                target:add("deps", dep)
            end
        end)
        if (opts["enable_cpp"]) then 
            add_files(folder .. "/" .. name .. "/*.cpp")
        end
        if (opts["enable_cu"]) then 
            add_files(folder .. "/" .. name .. "/*.cu")
        end

        add_deps("external_doctest")
        add_includedirs("_framework")
        add_files("_framework/test_util.cpp")
    target_end()
end

sail_add_test("basic", "dummy", {}) -- basic feature for cpp
sail_add_test("basic", "stl", {}) -- STL feature for cpp
sail_add_test("basic", "advanced", {}) -- advanced feature for cpp

-- for TBB and embree
sail_add_test("cpu", "tbb", {
    "SailCPU"
})

-- for CUDA
if has_config("sail_enable_cuda") then 
    if has_config("sail_enable_cuda_tensor") then 
        sail_add_test("cut", "cublas", {"SailCuT"}, {
            enable_cu=true, 
            -- enable_cpp=true
        })
        sail_add_test("cut", "cudnn", {"SailCuT"}, {
            enable_cu=true,
            -- enable_cpp=true
        })
        sail_add_test("cut", "cutlass", {"SailCuT"}, {
            enable_cu=true,
            -- enable_cpp=true
        })
    end
end