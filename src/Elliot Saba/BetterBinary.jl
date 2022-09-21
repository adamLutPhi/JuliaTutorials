"""Binary Builders 
JuliaCon2018 - 
Video: https://www.youtube.com/watch?v=2e0PBGSaQaI&ab_channel=TheJuliaProgrammingLanguage
Credits: Elliot `Eli` Saba @staticfloat (thank you so much!) :D 

Q.How are we gonna do that?
(I like the way he's seperating topics in)
1. Unpack compiled taballs
1.2. No:
1.2.1. package managers 
1.2.2. sudo 
1.2.3. Compilation 
1.2.4. NO STATE! 

2.Build Tarballs
2.1.How: using cross-compilation Environment 
2.1.1 Linux based: with:
2.1.1.1. gcc 
2.1.1.2. gfortran 
2.1.1.3. clang 

2.2 Corss compilers with everybody (so that performance gotta be the same )
Note: built for all targets 
2.2.1. windows 
2.2.2. arm64

2.3 Run inside a Cross Compilation Environment
2.3.1 Buld `all` the things - Anywhere
me: with its own special flavors & Recipies

"""
#=Strategy:
we'll `bake` a little Linux image on any linux or mac machine 
Juliacon2018 - Elliot Saba 
How: prerequisites: activate current directory and add package BinaryBuilder

1. using BinaryBuilder 
2. BinaryBuilder.runshell() 

q.

windows -> admin prevelages 

either findout chmod and try back on win 
    or run in mac 

#waiting to be consumed  on mac ... 
#1. requires only mac  (Ubuntu references  ) #####requires focal-updates" - Ubuntu Packages # Focal Fossa (Ubuntu) # https://wiki.ubuntu.com/FocalFossa/ReleaseNotes
2. remove
strategy: 
either load everything into project toml 
=#

    ENV["XDG_DATA_DIRS"] = join(filter(x -> x !== nothing, [
         #   dirname(adwaita_icons_dir), # 
         #   dirname(hicolor_icons_dir),
            joinpath(dirname(GTK3_jll.libgdk3_path::String), "..", "share"), #same problem as in Gui.jl 
            get(ENV, "XDG_DATA_DIRS", nothing)::Union{String,Nothing},#GTK3_jll not defined 
        ]), Sys.iswindows() ? ";" : ":")
#=
=#
#=
TODO: change currrent_pwd() to artifact_hash
# artifact_hash isa notdefined
loaders_cache_hash = artifact_hash(loaders_cache_name, mutable_artifacts_toml)
loaders_dir_hash = artifact_hash(loaders_dir_name, mutable_artifacts_toml)
# Install Ubuntu on WSL2 on Windows 10
# https://aka.ms/wslstore
# install Ubuntu on  
https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-10#3-download-ubuntu
 
LoadError: UndefVarError: create_artifact not definedStacktrace:

=#
mutable_artifacts_toml = joinpath(dirname(@__DIR__), "MutableArtifacts.toml")
loaders_cache_name = "gdk-pixbuf-loaders-cache"
#loaders_cache_hash = artifact_hash(loaders_cache_name, mutable_artifacts_toml)
loaders_dir_name = "gdk-pixbuf-loaders-dir"
#loaders_dir_hash = artifact_hash(loaders_dir_name, mutable_artifacts_toml)

# Copy loaders into a directory
loaders_dir_hash = create_artifact() do art_dir
    loaders_dir = mkdir(joinpath(art_dir, "loaders_dir"))
    pixbuf_loaders = joinpath.(gdk_pixbuf_loaders_dir, readdir(gdk_pixbuf_loaders_dir))
    push!(pixbuf_loaders, Librsvg_jll.libpixbufloader_svg)
    cp.(pixbuf_loaders, joinpath.(loaders_dir, basename.(pixbuf_loaders)))
end

loaders_dir = joinpath(artifact_path(loaders_dir_hash), "loaders_dir")
# Pkg removes "execute" permissions on Windows
if Sys.iswindows() && chmod(artifact_path(loaders_dir_hash), 0o755; recursive = true)#grant access to msft



end
using BinaryBuilder
BinaryBuilder.runshell()  