function add_demo_cut(name)
    target("demo_cut_" .. name)
        set_kind("binary")
        add_files(name .. ".cu")
        add_deps("SailCuT")
    target_end()
end 

add_demo_cut("00.basic_index")