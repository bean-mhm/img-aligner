cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

project(img-aligner VERSION 1.0)

# main executable
file(GLOB_RECURSE SRC_CPP_FILES src/*.cpp)
file(GLOB_RECURSE SRC_C_FILES src/*.c)
add_executable(img-aligner ${SRC_CPP_FILES} ${SRC_C_FILES})
set_target_properties(img-aligner PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<0:>")
set_target_properties(img-aligner PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<0:>")

# add compile definition for debug builds
target_compile_definitions(img-aligner PRIVATE
    $<$<CONFIG:Debug>:
    DEBUG_BUILD=1
    >
)

# compiler and linker flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(img-aligner PRIVATE -static -static-libgcc -static-libstdc++ -mconsole)
    target_link_options(img-aligner PRIVATE -static -static-libgcc -static-libstdc++ -mconsole)
endif()

# include dirs
target_include_directories(img-aligner PUBLIC src/lib)

# Vulkan
find_package(Vulkan REQUIRED)
target_include_directories(img-aligner PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(img-aligner PUBLIC ${Vulkan_LIBRARIES})

# GLFW
set(GLFW_LIBRARY_TYPE "STATIC")
add_subdirectory(thirdparty/glfw)
target_include_directories(img-aligner PUBLIC ${GLFW_SOURCE_DIR}/include)
target_link_libraries(img-aligner PUBLIC glfw)

# OpenEXR
add_subdirectory(thirdparty/openexr)
set(OPENEXR_INSTALL off)
set(OPENEXR_INSTALL_TOOLS off)
set(OPENEXR_BUILD_TOOLS off)
set(OPENEXR_BUILD_EXAMPLES off)
target_include_directories(img-aligner PUBLIC thirdparty/openexr/src/lib)
target_link_libraries(img-aligner PUBLIC OpenEXR)

# Native File Dialog Extended
add_subdirectory(thirdparty/nativefiledialog-extended)
target_include_directories(img-aligner PUBLIC thirdparty/nativefiledialog-extended/src/include)
target_link_libraries(img-aligner PRIVATE nfd)

# FreeType (needed for Dear ImGUI)
add_subdirectory(thirdparty/freetype)
target_include_directories(img-aligner PUBLIC thirdparty/freetype/include)
target_link_libraries(img-aligner PUBLIC freetype)

# Windows API
if(MSVC OR MSYS OR MINGW OR WIN32)
    target_link_libraries(img-aligner PUBLIC
        kernel32
        user32
        gdi32
        dwmapi
        winspool
        comdlg32
        advapi32
        shell32
        ole32
        oleaut32
        uuid
        odbc32
        odbccp32
    )
endif()

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

# figure out recursive directory copy command in the current platform
if(MSVC OR MSYS OR MINGW OR WIN32)
    set(CMD_RCOPY_PREFIX "xcopy")
    set(CMD_RCOPY_SUFFIX /s)
    list(APPEND COPY_COMMAND_SUFFIX /y)
else()
    set(CMD_RCOPY_PREFIX "cp -ra")
    set(CMD_RCOPY_SUFFIX "")
endif()

# compile shaders
set(GLSLC_PATH "C:/Program Files/VulkanSDK/1.3.250.1/Bin/glslc.exe")
mkdir_if_needed("${CMAKE_BINARY_DIR}/bin/shaders")
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND echo compiling shaders
    COMMAND "${GLSLC_PATH}" -fshader-stage=vertex "${CMAKE_SOURCE_DIR}/shaders/fullscreen_quad_vert.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/fullscreen_quad_vert.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=vertex "${CMAKE_SOURCE_DIR}/shaders/grid_warp_pass_vert.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/grid_warp_pass_vert.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/grid_warp_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/grid_warp_pass_frag.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/difference_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/difference_pass_frag.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/cost_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/cost_pass_frag.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/ui_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/ui_pass_frag.spv"
    COMMAND echo done compiling shaders
)

# copy resource files (make sure source directories have /. at the end)
mkdir_if_needed("${CMAKE_BINARY_DIR}/bin/licenses")
mkdir_if_needed("${CMAKE_BINARY_DIR}/bin/fonts")
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND echo copying resource files
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/licenses/.\" \"${CMAKE_BINARY_DIR}/bin/licenses\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/fonts/.\" \"${CMAKE_BINARY_DIR}/bin/fonts\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/imgui.ini" "${CMAKE_BINARY_DIR}/bin/imgui.ini"
    COMMAND echo done copying resource files
)

# find required runtime DLLs
find_file(LIBGCC_DLL libgcc_s_seh-1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(LIBSTDCXX_DLL libstdc++-6.dll PATHS ${CMAKE_CXX_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(LIBWINPTHREAD_DLL libwinpthread-1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(LIBBZ2_DLL libbz2-1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(ZLIB_DLL zlib1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)

# copy required DLLs to the output directory
add_custom_command(TARGET img-aligner POST_BUILD
    COMMAND echo copying runtime DLLs

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    # ${LIBGCC_DLL} (these are statically linked)
    # ${LIBSTDCXX_DLL} (these are statically linked)
    # ${LIBWINPTHREAD_DLL} (these are statically linked)
    ${LIBBZ2_DLL}
    ${ZLIB_DLL}
    ${CMAKE_BINARY_DIR}/bin

    COMMAND echo done copying runtime DLLs
)
