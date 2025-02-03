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

# add compile definition for debug builds
target_compile_definitions(img-aligner PRIVATE
    $<$<CONFIG:Debug>:
    DEBUG_BUILD=1
    >
)

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

# Imath
set(Imath_DIR Imath)
set(Imath_ROOT Imath)
add_subdirectory(Imath)
target_include_directories(img-aligner PUBLIC Imath/src)
target_link_libraries(img-aligner PUBLIC Imath)

# expat
set(expat_DIR libexpat/expat)
set(expat_ROOT libexpat/expat)
add_subdirectory(libexpat/expat)
target_include_directories(img-aligner PUBLIC libexpat/expat/lib)
target_link_libraries(img-aligner PUBLIC expat)

# minizip-ng
set(minizip-ng_DIR minizip-ng)
set(minizip-ng_ROOT minizip-ng)
add_subdirectory(minizip-ng)
target_include_directories(img-aligner PUBLIC minizip-ng)
target_link_libraries(img-aligner PUBLIC minizip-ng)

# pystring
set(pystring_DIR pystring)
set(pystring_ROOT pystring)
add_subdirectory(pystring)
target_include_directories(img-aligner PUBLIC pystring)
target_link_libraries(img-aligner PUBLIC pystring)

# yaml-cpp
set(yaml-cpp_DIR yaml-cpp)
set(yaml-cpp_ROOT yaml-cpp)
add_subdirectory(yaml-cpp)
target_include_directories(img-aligner PUBLIC yaml-cpp/include)
target_link_libraries(img-aligner PUBLIC yaml-cpp)

# zlib
set(zlib_DIR zlib)
set(zlib_ROOT zlib)
add_subdirectory(zlib)
target_include_directories(img-aligner PUBLIC zlib)
target_link_libraries(img-aligner PUBLIC zlib)

# OpenEXR
set(openexr_DIR openexr)
set(openexr_ROOT openexr)
add_subdirectory(openexr)
target_include_directories(img-aligner PUBLIC openexr/src/lib/OpenEXR)
target_link_libraries(img-aligner PUBLIC OpenEXR)

# OpenColorIO
set(OCIO_BUILD_APPS OFF)
set(OCIO_BUILD_TESTS OFF)
set(OCIO_BUILD_GPU_TESTS OFF)
set(OCIO_BUILD_PYTHON OFF)
add_subdirectory(OpenColorIO)
target_include_directories(img-aligner PUBLIC OpenColorIO/include)
target_link_libraries(img-aligner PUBLIC OpenColorIO)

# macro to make a directory if it doesn't exist (at build time)
if(MSVC OR MSYS OR MINGW OR WIN32)
    macro(mkdir_if_needed dir)
        add_custom_command(TARGET img-aligner
            POST_BUILD
            COMMAND if not exist \"${dir}\" mkdir \"${dir}\"
        )
    endmacro()
else()
    macro(mkdir_if_needed dir)
        add_custom_command(TARGET img-aligner
            POST_BUILD
            COMMAND if \[ ! -d \"${dir}\" \]; then mkdir \"${dir}\" fi
        )
    endmacro()
endif()

# figure out the copy command in the current platform
if(MSVC OR MSYS OR MINGW OR WIN32)
    set(COPY_COMMAND_PREFIX "xcopy")
    set(COPY_COMMAND_SUFFIX /s)
    list(APPEND COPY_COMMAND_SUFFIX /y)
else()
    set(COPY_COMMAND_PREFIX "cp -ra")
    set(COPY_COMMAND_SUFFIX "")
endif()

# compile shaders
set(GLSLC_PATH "C:/Program Files/VulkanSDK/1.3.250.1/Bin/glslc.exe")
mkdir_if_needed("${CMAKE_BINARY_DIR}/bin/shaders")
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND echo compiling shaders
    COMMAND "${GLSLC_PATH}" -fshader-stage=vertex "${CMAKE_SOURCE_DIR}/shaders/vert.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/vert.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/frag.spv"
    COMMAND echo done compiling shaders
)

# copy resource files (make sure source directories have /. at the end)
mkdir_if_needed("${CMAKE_BINARY_DIR}/bin/licenses")
mkdir_if_needed("${CMAKE_BINARY_DIR}/bin/fonts")
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND echo copying resource files
    COMMAND ${COPY_COMMAND_PREFIX} \"${CMAKE_SOURCE_DIR}/licenses/.\" \"${CMAKE_BINARY_DIR}/bin/licenses\" ${COPY_COMMAND_SUFFIX}
    COMMAND ${COPY_COMMAND_PREFIX} \"${CMAKE_SOURCE_DIR}/fonts/.\" \"${CMAKE_BINARY_DIR}/bin/fonts\" ${COPY_COMMAND_SUFFIX}
    COMMAND echo done copying resource files
)
