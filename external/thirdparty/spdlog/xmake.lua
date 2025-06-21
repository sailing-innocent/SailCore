target("external_spdlog")
    set_kind("static")
    add_rules("sail.predef")
    add_includedirs("include", {public = true})
    add_files("src/*.cpp")

    on_load(function(target)
        target:add("defines", "SPDLOG_COMPILED_LIB", {
            public = true
        })
    end)
target_end()
