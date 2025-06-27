function add_demo_cpp(name, deps)
    target("demo_cpu_" .. name)
        set_kind("binary")
        add_files(name .. ".cpp")
        set_group("03.demos/cpu")
        add_deps(deps)
        add_deps("SailCPU")
        add_deps("external_argparse")
    target_end()
end 

add_demo_cpp("00.ppl") -- Parallel Patterns Library
add_demo_cpp("01.use_embree") -- Intel Embree