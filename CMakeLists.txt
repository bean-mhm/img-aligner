cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

project(img-aligner VERSION 1.0)

# see if we're building for Windows
if (MSVC OR MSYS OR MINGW OR WIN32)
    set(WINDOWS ON)
else()
    set(WINDOWS OFF)
endif()

# figure out the executable file extension in the current platform
if(WINDOWS)
    set(EXEC_FILE_EXT ".exe")
elseif(APPLE)
    set(EXEC_FILE_EXT ".app")
else()
    set(EXEC_FILE_EXT "")
endif()

# main executable
file(GLOB_RECURSE SRC_CPP_FILES src/*.cpp)
file(GLOB_RECURSE SRC_C_FILES src/*.c)
add_executable(img-aligner ${SRC_CPP_FILES} ${SRC_C_FILES})

# set binary output directory
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

# setup icon and resources
if(WINDOWS)
    set(RC_FILE ${CMAKE_BINARY_DIR}/resources/app.rc)
    file(
        WRITE
        ${RC_FILE}
        "IDI_ICON1 ICON DISCARDABLE \"${CMAKE_SOURCE_DIR}/resources/icon.ico\"\n"
        "GLFW_ICON ICON \"${CMAKE_SOURCE_DIR}/resources/icon.ico\"\n"
    )
    target_sources(img-aligner PRIVATE ${RC_FILE})
elseif(APPLE)
    # IDK if this will work. I haven't tested it. I got it from AI.
    set(MACOSX_BUNDLE_ICON_FILE ${CMAKE_SOURCE_DIR}/resources/icon.icns)
    set_source_files_properties(${MACOSX_BUNDLE_ICON_FILE} PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
    set_target_properties(img-aligner PROPERTIES
        MACOSX_BUNDLE TRUE
        MACOSX_BUNDLE_ICON_FILE ${MACOSX_BUNDLE_ICON_FILE}
    )
    target_sources(img-aligner PRIVATE ${MACOSX_BUNDLE_ICON_FILE})
endif()

# Vulkan
find_package(Vulkan REQUIRED)
target_include_directories(img-aligner PUBLIC ${Vulkan_INCLUDE_DIRS})
target_link_libraries(img-aligner PUBLIC ${Vulkan_LIBRARIES})

# GLFW
set(GLFW_LIBRARY_TYPE "STATIC" CACHE STRING "GLFW library type")
add_subdirectory(thirdparty/glfw)
target_include_directories(img-aligner PUBLIC ${GLFW_SOURCE_DIR}/include)
target_link_libraries(img-aligner PUBLIC glfw)

# OpenEXR
set(OPENEXR_BUILD_TOOLS OFF CACHE BOOL "disable unwanted OpenEXR binaries")
set(OPENEXR_BUILD_EXAMPLES OFF CACHE BOOL "disable unwanted OpenEXR binaries")
add_subdirectory(thirdparty/openexr)
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

# fmt
add_subdirectory(thirdparty/fmt)
target_include_directories(img-aligner PUBLIC thirdparty/fmt/include)
target_link_libraries(img-aligner PRIVATE fmt)

# Windows API
if(WINDOWS)
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
if(WINDOWS)
    macro(mkdir_if_needed_post_build dir)
        add_custom_command(TARGET img-aligner
            POST_BUILD
            COMMAND if not exist \"${dir}\" mkdir \"${dir}\"
            DEPENDS ALWAYS
        )
    endmacro()
else()
    macro(mkdir_if_needed_post_build dir)
        add_custom_command(TARGET img-aligner
            POST_BUILD
            COMMAND if \[ ! -d \"${dir}\" \]; then mkdir \"${dir}\" fi
            DEPENDS ALWAYS
        )
    endmacro()
endif()

# figure out recursive directory copy command in the current platform
if(WINDOWS)
    set(CMD_RCOPY_PREFIX "xcopy")
    set(CMD_RCOPY_SUFFIX /s)
    list(APPEND COPY_COMMAND_SUFFIX /y)
else()
    set(CMD_RCOPY_PREFIX "cp -ra")
    set(CMD_RCOPY_SUFFIX "")
endif()

# post build: compile shaders
set(GLSLC_PATH "C:/Program Files/VulkanSDK/1.3.250.1/Bin/glslc.exe")
mkdir_if_needed_post_build("${CMAKE_BINARY_DIR}/bin/shaders")
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo compiling shaders
    COMMAND "${GLSLC_PATH}" -fshader-stage=vertex "${CMAKE_SOURCE_DIR}/shaders/fullscreen_quad_vert.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/fullscreen_quad_vert.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=vertex "${CMAKE_SOURCE_DIR}/shaders/grid_warp_pass_vert.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/grid_warp_pass_vert.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/grid_warp_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/grid_warp_pass_frag.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/difference_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/difference_pass_frag.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/cost_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/cost_pass_frag.spv"
    COMMAND "${GLSLC_PATH}" -fshader-stage=fragment "${CMAKE_SOURCE_DIR}/shaders/ui_pass_frag.glsl" -o "${CMAKE_BINARY_DIR}/bin/shaders/ui_pass_frag.spv"
    COMMAND ${CMAKE_COMMAND} -E echo done compiling shaders
    DEPENDS ALWAYS
)

# post build: copy resource files (make sure source directories have /. at the end)
mkdir_if_needed_post_build("${CMAKE_BINARY_DIR}/bin/licenses")
mkdir_if_needed_post_build("${CMAKE_BINARY_DIR}/bin/fonts")
mkdir_if_needed_post_build("${CMAKE_BINARY_DIR}/bin/demo")
mkdir_if_needed_post_build("${CMAKE_BINARY_DIR}/bin/docs")
mkdir_if_needed_post_build("${CMAKE_BINARY_DIR}/bin/images")
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo copying resource files
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/licenses/.\" \"${CMAKE_BINARY_DIR}/bin/licenses\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/fonts/.\" \"${CMAKE_BINARY_DIR}/bin/fonts\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/demo/.\" \"${CMAKE_BINARY_DIR}/bin/demo\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/docs/.\" \"${CMAKE_BINARY_DIR}/bin/docs\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/images/.\" \"${CMAKE_BINARY_DIR}/bin/images\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/LICENSE.txt" "${CMAKE_BINARY_DIR}/bin/LICENSE.txt"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/README.md" "${CMAKE_BINARY_DIR}/bin/README.md"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_SOURCE_DIR}/imgui.ini" "${CMAKE_BINARY_DIR}/bin/imgui.ini"
    COMMAND ${CMAKE_COMMAND} -E echo done copying resource files
    DEPENDS ALWAYS
)

# find required runtime DLLs
find_file(LIBGCC_DLL libgcc_s_seh-1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(LIBSTDCXX_DLL libstdc++-6.dll PATHS ${CMAKE_CXX_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(LIBWINPTHREAD_DLL libwinpthread-1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(LIBBZ2_DLL libbz2-1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)
find_file(ZLIB_DLL zlib1.dll PATHS ${CMAKE_C_COMPILER_PATH} PATH_SUFFIXES ../bin REQUIRED)

# post build: copy required DLLs to the output directory
add_custom_command(TARGET img-aligner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo copying runtime DLLs

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    # ${LIBGCC_DLL} (these are statically linked)
    # ${LIBSTDCXX_DLL} (these are statically linked)
    # ${LIBWINPTHREAD_DLL} (these are statically linked)
    ${LIBBZ2_DLL}
    ${ZLIB_DLL}
    ${CMAKE_BINARY_DIR}/bin

    COMMAND ${CMAKE_COMMAND} -E echo done copying runtime DLLs

    DEPENDS ALWAYS
)
