includes("doctest")
target("external_imgui")
    set_kind("phony")
    add_packages("imgui", { public = true})
target_end()

if get_config("sail_enable_gl") then 
    includes("glad")
    target("external_glm")
        set_kind("phony")
        add_packages("glm", { public = true})
    target_end()
end 

includes("stb_util")
includes("imath")
includes("alembic")
includes("tiny_obj_loader_util")
includes("tiny_gltf_loader_util")
includes("lua")
includes("happly")
includes("spdlog")
includes("argparse")
