target("vulkan_shaders")
    set_kind("object")
    add_rules("utils.glsl2spv", {outputdir="bin/assets/shaders/vulkan" }) 
    add_files("*.frag", "*.vert")
target_end()