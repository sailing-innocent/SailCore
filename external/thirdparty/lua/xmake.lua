-- version 5.4.7
target("lua")
    set_kind("shared")
    add_rules("sail.predef")
    add_defines(
       "LUA_BUILD_AS_DLL", 
       "LUA_OPNAMES", 
       "LUA_WITH_COUT", 
       "LUA_USES__PRINT"
    )
    add_includedirs("src", {public=true})
    add_files("src/*.c")
    remove_files("src/luac.c", "src/lua.c")
    -- lua.c is the interpreter and luac.c is the compiler
target_end()