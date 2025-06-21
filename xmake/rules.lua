rule("sail.cuda")
    on_config(function(target)
    end)
    on_load(function(target)
        target:set("cuda.rdc", false)
        local cuda_path = os.getenv("CUDA_PATH")
        if cuda_path then
            target:add("sysincludedirs", path.join(cuda_path, "include"), {public=true})
            target:add("linkdirs", path.join(cuda_path, "lib/x64/"), {public=true})
            target:add("links", "nvrtc", "cudart", "cuda", "cublas", {public=true})
        else
            target:set("enabled", false)
            print("$env:CUDA_PATH is not set")
            return
        end
        if is_plat("windows") then
            target:add("defines", "NOMINMAX", "UNICODE", {public=true})
            target:add("syslinks", "Cfgmgr32", "Advapi32", {public=true})
        end
    end)

    after_build(function(target)
        local cuda_path = os.getenv("CUDA_PATH")
        if cuda_path then
            shared_files = os.match(path.join(cuda_path, "bin/*.dll"))
            -- if shared files not exists(targetdir()), copy them
            local shared_file_names = {}
            for _, shared_file in ipairs(shared_files) do
                table.insert(shared_file_names, path.filename(shared_file))
            end
            for _, shared_file_name in ipairs(shared_file_names) do
                if not os.isfile(path.join(target:targetdir(), shared_file_name)) then
                    os.cp(path.join(cuda_path, "bin", shared_file_name), target:targetdir())
                    print("copy " .. shared_file_name .. " to " .. target:targetdir())
                end
            end
            local devrt_lib_name = "cudadevrt.lib"
            local dst_file = path.join(target:targetdir(), devrt_lib_name)
            if not os.isfile(dst_file) then
                os.cp(path.join(cuda_path, "lib/x64", devrt_lib_name), target:targetdir())
            end
        end
    end)
rule_end()

rule("sail.cudnn")
    add_deps("sail.cuda")
    on_load(function(target)
        local static_lib_names = {
            "cudnn"
        }
        -- local static_lib_names = {
        --     "cudnn_ops",
        --     "cudnn_adv",
        --     "cudnn_cnn",
        --     "cudnn_graph",
        --     "cudnn_heuristic",
        --     "cudnn_engines_runtime_compiled",
        --     "cudnn_engines_precompiled"
        -- }
        local cuda_version = "12.8"
        local cudnn_path = os.getenv("CUDNN_PATH")
        if not cudnn_path then
            print("$env:CUDNN_PATH is not set")
            return
        end
        if not cuda_version then
            print("cuda_version is not set")
            return
        end
        local cudnn_lib_path = path.join(cudnn_path, "lib", cuda_version, "x64")

        local cudnn_include_path = path.join(cudnn_path, "include", cuda_version)
        target:add("sysincludedirs", cudnn_include_path, {public=true})
        target:add("linkdirs", cudnn_lib_path, {public=true})
        target:add("links", static_lib_names, {public=true})

    end)
    after_build(function(target)
        local cudnn_path = os.getenv("CUDNN_PATH")
        local cuda_version = "12.8"
        local cudnn_dll_path = path.join(cudnn_path, "bin", cuda_version)
        local dll_files = os.match(path.join(cudnn_dll_path, "*.dll"))
        for _, dll_file in ipairs(dll_files) do
            if not os.isfile(path.join(target:targetdir(), path.filename(dll_file))) then
                os.cp(dll_file, target:targetdir())
                print("copy " .. dll_file .. " to " .. target:targetdir())
            end
        end
    end)
rule_end()


rule("sail.optix")
    add_deps("sail.cuda")
    on_load(function(target)
        -- local optix_path = "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0"
        local optix_path = os.getenv("CUDA_OPTIX_PATH")
        if not optix_path then
            print("$env:CUDA_OPTIX_PATH is not set")
            return
        end
        target:add("sysincludedirs", path.join(optix_path, "include"), {public=true})
        -- link to cudart64_12
        target:add("cuflags", "-arch=sm_60")
    end)
    after_build(function(target)
    end)
rule_end()

rule("sail.tensorrt")
    add_deps("sail.cuda")
    on_load(function(target)
        local tensorrt_path = os.getenv("TENSORRT_PATH")
        if not tensorrt_path then
            print("$env:TENSORRT_PATH is not set")
            return
        end
        target:add("sysincludedirs", path.join(tensorrt_path, "include"), {public=true})
        target:add("linkdirs", path.join(tensorrt_path, "lib"), {public=true})
        target:add("links", "nvinfer_10", "nvinfer_plugin_10", {public=true})
    end)

    after_build(function(target)
        local tensorrt_path = os.getenv("TENSORRT_PATH")
        if not tensorrt_path then
            print("$env:TENSORRT_PATH is not set")
            return
        end
        local dll_files = os.match(path.join(tensorrt_path, "lib", "*.dll"))
        for _, dll_file in ipairs(dll_files) do
            if not os.isfile(path.join(target:targetdir(), path.filename(dll_file))) then
                os.cp(dll_file, target:targetdir())
                print("copy " .. dll_file .. " to " .. target:targetdir())
            end
        end
    end)
rule_end()

rule("sail.llvm")
    on_load(function(target)
        local llvm_path = get_config("llvm_path")
        local llvm_include_dir = llvm_path .. "/include"
        local llvm_lib_dir = llvm_path .. "/lib"
        local libs = {}
        local p = llvm_path .. "/lib/*.lib"
        for __, filepath in ipairs(os.files(p)) do
            local basename = path.basename(filepath)
            table.insert(libs, basename)
        end
        target:add("linkdirs", llvm_lib_dir, {public=true})
        target:add("links", libs, {public=true})
        target:add("includedirs", llvm_include_dir, { public = true})
    end)
rule_end()

rule("sail.win")
on_load(function(target)
    target:add("syslinks", "user32", "gdi32", "shell32", "advapi32", "ole32", "oleaut32", "uuid", "comdlg32", "comctl32", "kernel32", {public=true})
    if is_mode("debug") then
        target:add("syslinks", "ucrtd", "msvcrtd", "vcruntimed", {public=true})
    elseif is_mode("releasedbg") then
        target:add("syslinks", "ucrtd", "msvcrtd", "vcruntimed", {public=true})
    else
        target:add("syslinks", "ucrt", "msvcrt", "vcruntime", {public=true})
    end
end)
rule_end()

function LIBRARY_DEPENDENCY(dep, version, settings)
    add_deps(dep, {public=true})
    add_values(dep .. ".version", version)
end

function PUBLIC_DEPENDENCY(dep, version, settings)
    add_deps(dep, {public=true})
    add_values("sail.module.public_dependencies", dep)
    add_values(dep .. ".version", version)
end

rule("sail.predef")
    on_load(function(target)
        target:add("defines", "_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR", {public=true})
        -- see https://developercommunity.visualstudio.com/t/All-my-std::unique_lock-crashed-after-th/10665376?space=41&sort=newest&viewtype=all
        -- https://www.soinside.com/question/VTab6FvWLARJWzigEhyBeC
        -- https://github.com/microsoft/STL/wiki/Changelog mutex constexpr constructor
    end)
rule_end()

rule("sail.component")
    add_deps("sail.predef")
    on_config(function (component, opt)
        import("core.project.project")
        local owner_name = component:extraconf("rules", "sail.component", "owner")
        local owner = project.target(owner_name)
        -- insert owner's include dirs
        for _, owner_inc in pairs(owner:get("includedirs")) do
            component:add("includedirs", owner_inc, {public = true})
        end
        local owner_api = owner:extraconf("rules", "sail.dynamic_module", "api") or owner:extraconf("rules", "sail.static_module", "api")
        -- import api from owner
        component:add("defines", owner_api.."_API=SAIL_IMPORT", owner_api.."_LOCAL=error")
    end)
rule_end()

function STATIC_COMPONENT(name, owner, settings)
    target(owner)
        add_deps(name, { public = opt and opt.public or true })
    target_end()

    target(name)
        set_group("01.modules/"..owner.."/components")
        add_rules("sail.component", { owner = owner })
        set_kind("static")    
end

rule("sail.module")
    add_deps("sail.predef")
    on_load(function (target, opt)
        if not os.isdir(target:targetdir()) then
            os.mkdir(target:targetdir())
        end
        target:set("rtti", true)
        target:add("vectorexts", "avx", "avx2")
    end)
    on_config(function(target)
    end)
    after_build(function(target)
    end)
rule_end()

rule("sail.static_library")
    on_load(function (target, opt)
        target:set("kind", "static")
    end)
rule_end()

rule("sail.static_module")
    add_deps("sail.module")
    add_deps("sail.static_library")
    on_load(function(target, opt)
        local api = target:extraconf("rules", "sail.static_module", "api")
        target:add("defines", api.."_API", {public=true})
        target:add("defines", api.."_STATIC", {public=true})
        target:add("defines", api.."_IMPL")
    end)
rule_end()

rule("sail.dynamic_module")
    add_deps("sail.module")
    on_load(function(target, opt)
        local api = target:extraconf("rules", "sail.dynamic_module", "api")
        local version = target:extraconf("rules", "sail.dynamic_module", "version")
        target:add("defines", api.."_API=SAIL_IMPORT", {public=true})
        target:add("defines", api.."_API=SAIL_EXPORT", {public=false})
    end)
rule_end()

function SHARED_MODULE(name, api, version, opt) 
    target(name)
        set_kind("shared")
        set_languages("clatest", "c++20")
        set_exceptions("cxx")
        set_group("01.modules/" .. name)
        add_rules("sail.dynamic_module", {api=api, version=version})

end


