cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

project(tglang VERSION 1.0.0 LANGUAGES C)
set(CMAKE_C_STANDARD 11)

option(EMBED_WEIGHTS "If ON, then models' weights will be embed into binaries" OFF)

# Check if a certain condition is met
if(EMBED_WEIGHTS)
    # Code to be executed when the condition is true
    add_definitions(-DEMBED_WEIGHTS)
endif()

# tokenizer, gru and svc can compile for a long time (if embedding is on)
# so we compile object files for them once and then use them multiple times
add_library(tokenizer_obj OBJECT src/tokenizer.c)
add_library(gru_obj OBJECT src/gru.c)
add_library(svc_obj OBJECT src/svc.c)

add_executable(test src/test.c
                    src/matrix.c
                    src/tglang.c
                    $<TARGET_OBJECTS:tokenizer_obj>
                    $<TARGET_OBJECTS:gru_obj>
                    $<TARGET_OBJECTS:svc_obj>)

add_executable(embed_weights src/embed_weights.c
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
target_link_libraries(test m)
target_link_libraries(embed_weights m)
target_link_libraries(gru m)
target_link_libraries(tglang m)

target_compile_options(test PUBLIC "-mavx2" "-ffast-math")
target_compile_options(embed_weights PUBLIC "-mavx2")
target_compile_options(tglang PUBLIC "-fPIC" "-mavx2" "-ffast-math")
target_compile_options(tokenizer_obj PUBLIC "-fPIC" "-mavx2")
target_compile_options(gru PUBLIC "-fPIC" "-mavx2" "-ffast-math")
target_compile_options(gru_obj PUBLIC "-fPIC" "-mavx2" "-ffast-math")
target_compile_options(svc_obj PUBLIC "-fPIC" "-mavx2" "-ffast-math")
