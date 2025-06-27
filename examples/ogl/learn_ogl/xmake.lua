target("learn_ogl_util")
    set_kind("static")
    add_files("util.cpp")
    add_packages("glfw", {public = true})
    add_deps(
        "external_glad", 
        "external_stb_util", 
        "external_glm",
    {public = true})
target_end()

function add_ogl_demo(name, deps) 
    target("learn_ogl_" .. name)
        set_kind("binary")
        add_deps(deps)
        add_files(name .. ".cpp")
        add_deps("learn_ogl_util")
    target_end()
end

add_ogl_demo("00.plain_window")
add_ogl_demo("01.triangle")
add_ogl_demo("02.rectangle")
add_ogl_demo("03.texture", "external_argparse")
add_ogl_demo("04.transform", "external_argparse")
add_ogl_demo("05.coord", "external_argparse")
add_ogl_demo("06.camera", "external_argparse")