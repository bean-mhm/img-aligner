cmake_minimum_required(VERSION 3.24)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

project(img-aligner VERSION 1.0)

# see if we're building for Windows
set(WINDOWS MSVC OR MSYS OR MINGW OR WIN32)

# figure out the executable file extension in the current platform
if(WINDOWS)
    set(EXEC_FILE_EXT ".exe")
elseif(APPLE)
    set(EXEC_FILE_EXT ".app")
else()
    set(EXEC_FILE_EXT "")
endif()

if(WINDOWS)
    # compile resources (also containing icon). this will run at configure time,
    # so if the resources change we should reconfigure CMake.
    find_program(WINDRES_EXECUTABLE windres REQUIRED)
    execute_process(
        COMMAND ${WINDRES_EXECUTABLE} app.rc -o ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.o
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE WINDRES_RESULT
        ERROR_VARIABLE WINDRES_ERROR
        # OUTPUT_VARIABLE UNUSED
        # OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(NOT WINDRES_RESULT EQUAL 0)
        message(FATAL_ERROR "failed to compile resources: ${WINDRES_ERROR}")
    endif()

    set(COMPILED_RESOURCE_FILE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.o)
else()
    message(WARNING "compiling resources and icons isn't yet implemented on non-Windows platforms")
    set(COMPILED_RESOURCE_FILE "")
endif()

# main executable
file(GLOB_RECURSE SRC_CPP_FILES src/*.cpp)
file(GLOB_RECURSE SRC_C_FILES src/*.c)
add_executable(
    img-aligner
    ${SRC_CPP_FILES}
    ${SRC_C_FILES}
    ${COMPILED_RESOURCE_FILE}
)

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

# post build: clean up OpenEXR executables
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo performing post-build cleanup
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exr2aces${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrcheck${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrenvmap${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrheader${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrinfo${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrmakepreview${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrmaketiled${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrmanifest${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrmetrics${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrmultipart${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrmultiview${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/exrstdattr${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_BINARY_DIR}/bin/OpenEXRExamples${EXEC_FILE_EXT}"
    COMMAND ${CMAKE_COMMAND} -E echo done performing post-build cleanup
    DEPENDS ALWAYS
)

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
add_custom_command(TARGET img-aligner
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo copying resource files
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/licenses/.\" \"${CMAKE_BINARY_DIR}/bin/licenses\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/fonts/.\" \"${CMAKE_BINARY_DIR}/bin/fonts\"" ${CMD_RCOPY_SUFFIX}
    COMMAND ${CMD_RCOPY_PREFIX} "\"${CMAKE_SOURCE_DIR}/demo/.\" \"${CMAKE_BINARY_DIR}/bin/demo\"" ${CMD_RCOPY_SUFFIX}
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
