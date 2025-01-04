cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(img-aligner VERSION 1.0)

# main executable
file(GLOB_RECURSE SRC_CPP_FILES src/*.cpp)
file(GLOB_RECURSE SRC_C_FILES src/*.c)
add_executable(img-aligner ${SRC_CPP_FILES} ${SRC_C_FILES})
set_target_properties(img-aligner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<0:>")

# include dirs
target_include_directories(img-aligner PUBLIC src/lib)

# Vulkan
find_package(Vulkan REQUIRED)
target_include_directories(img-aligner PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(img-aligner PUBLIC ${Vulkan_LIBRARIES})

# GLFW
set(GLFW_LIBRARY_TYPE "STATIC")
add_subdirectory(glfw)
target_include_directories(img-aligner PUBLIC ${GLFW_SOURCE_DIR}/include)
target_link_libraries(img-aligner PUBLIC glfw)

# compile shaders
set(GLSLC_PATH "C:/Program Files/VulkanSDK/1.3.250.1/Bin/glslc.exe")
message("compiling shaders")
if(NOT EXISTS "${CMAKE_BINARY_DIR}/bin/shaders")
    add_custom_command(TARGET img-aligner
        POST_BUILD
        COMMAND if not exist \"${CMAKE_BINARY_DIR}/bin/shaders\" mkdir \"${CMAKE_BINARY_DIR}/bin/shaders\"
    )
    message("done compiling shaders")
endif()
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND "${GLSLC_PATH}" -fshader-stage=vertex "${CMAKE_SOURCE_DIR}/shaders/vert.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/vert.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/frag.spv"
)
