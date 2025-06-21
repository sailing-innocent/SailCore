SHARED_MODULE("SailCuT", "SAIL_CUT", engine_version)
    add_includedirs("include", { public = true })
    add_files("src/**.cu")
    add_files("src/**.cpp")
    add_deps("SailCu")
    add_deps("cutlass")
    add_rules("sail.cudnn")
    