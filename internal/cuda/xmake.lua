SHARED_MODULE("SailCu", "SAIL_CU", engine_version)
    add_includedirs("include", { public = true })
    add_rules("sail.cuda")
    add_rules("sail.optix")
    add_files("src/**.cu")
    -- add_files("src/**.cpp")
    add_deps("SailBase")
    -- add_deps("SailGL", { public = true }) -- for visualization
    -- add_packages("glm", { public = true })

