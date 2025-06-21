target("cutlass")
    set_kind("headeronly")
    add_headerfiles("include/**.hpp")
    add_includedirs("include", {public=true})
    add_rules("sail.cuda") -- cuda sdk
target_end()