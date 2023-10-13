cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES C)
set(CMAKE_C_STANDARD 11)

add_library(tokenizer_obj OBJECT src/tokenizer.c)
add_library(gru_obj OBJECT src/gru.c)
add_library(svc_obj OBJECT src/svc.c)

add_executable(main src/main.c
                    src/matrix.c
                    $<TARGET_OBJECTS:tokenizer_obj>
                    $<TARGET_OBJECTS:gru_obj>
                    $<TARGET_OBJECTS:svc_obj>)

add_library(tokenizer SHARED $<TARGET_OBJECTS:tokenizer_obj>)
add_library(gru SHARED $<TARGET_OBJECTS:gru_obj> src/matrix.c)
add_library(tglang SHARED src/tglang.c
                          src/matrix.c
                          $<TARGET_OBJECTS:tokenizer_obj>
                          $<TARGET_OBJECTS:gru_obj>
                          $<TARGET_OBJECTS:svc_obj>)

# link math.h lib
target_link_libraries(main m)
target_link_libraries(gru m)
target_link_libraries(tglang m)


if (CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(main PUBLIC "-mavx2" "-funroll-loops" "-ffast-math" "-O3" "-march=x86-64")
    target_compile_options(tglang PUBLIC "-fPIC" "-mavx2" "-funroll-loops" "-ffast-math" "-O3" "-march=x86-64")
    target_compile_options(gru PUBLIC "-fPIC" "-mavx2" "-funroll-loops" "-ffast-math" "-O3" "-march=x86-64")
    target_compile_options(tokenizer_obj PUBLIC "-fPIC" "-O3" "-march=x86-64")
    target_compile_options(gru_obj PUBLIC "-fPIC" "-mavx2" "-funroll-loops" "-ffast-math" "-O3" "-march=x86-64")
    target_compile_options(svc_obj PUBLIC "-fPIC" "-mavx2" "-O3" "-march=x86-64")
else()
    target_compile_options(main PUBLIC "-mavx2" "-g" "-O0")
    target_compile_options(tglang PUBLIC "-fPIC" "-mavx2" "-g" "-O0")
    target_compile_options(gru PUBLIC "-fPIC" "-mavx2" "-g" "-O0")
    target_compile_options(tokenizer_obj PUBLIC "-fPIC" "-g" "-O0")
    target_compile_options(gru_obj PUBLIC "-fPIC" "-mavx2" "-g" "-O0")
    target_compile_options(svc_obj PUBLIC "-fPIC" "-g" "-O0")
endif()
