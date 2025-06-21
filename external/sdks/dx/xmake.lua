target("dxsdk")
    set_kind("headeronly")
    add_headerfiles("include/**.h")
    add_includedirs("include", {public=true})
    add_rules("sail.win") -- windows sdk
    add_links("d3d12", "dxgi", "D3DCompiler", {public = true})
target_end()