target("terrain_assets")
    add_files("**.jpg", "**.png") -- textures
    add_files("**.obj", "**.mtl") -- models
    add_files("**.vert", "**.frag", "**.comp", "**.tcs", "**.tes")
    add_files("**.ttf") -- fonts
    add_rules("sail.asset")
target_end()