cmake_minimum_required(VERSION 3.6)
project(Projet_GPU)

link_directories(fichiers/lib)
link_directories(/usr/local/lib)

configure_file(fichiers/kernel/compute.cl kernel/compute.cl COPYONLY)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -g -Wall -Wno-unused-function -fopenmp")

set(SOURCE_FILES
    fichiers/include/compute.h
    fichiers/include/constants.h
    fichiers/include/debug.h
    fichiers/include/draw.h
    fichiers/include/error.h
    fichiers/include/graphics.h
    fichiers/include/ocl.h
    fichiers/src/compute.c
    fichiers/src/debug.c
    fichiers/src/draw.c
    fichiers/src/graphics.c
    fichiers/src/main.c
    fichiers/src/ocl.c)

add_executable(prog ${SOURCE_FILES})

target_link_libraries(prog m SDL2 SDL2_image visu OpenCL GL)

target_include_directories(prog PUBLIC fichiers/include)
target_include_directories(prog PUBLIC /usr/local/include/SDL2)
target_include_directories(prog PUBLIC /usr/include/SDL2)

target_compile_definitions(prog PUBLIC _REENTRANT)