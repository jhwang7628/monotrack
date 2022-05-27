cmake_minimum_required(VERSION 3.16)
project(detect)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
add_compile_options(
    -Wall
    -Wextra
    -O3
    -DNDEBUG
)

# Fetch opencv from source
set(FETCHCONTENT_BASE_DIR "${CMAKE_BINARY_DIR}/_deps" CACHE PATH "Directory under which to collect all populated content")


list(PREPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake/recipes/external/"
    "${PROJECT_SOURCE_DIR}/cmake/recipes/internal/"
)

include(opencv)

file(GLOB INC_FILES "*.h")
file(GLOB SRC_FILES "*.cpp")
add_executable(detect ${INC_FILES} ${SRC_FILES})

# top level include
target_include_directories(detect PRIVATE
    ${opencv_SOURCE_DIR}/include
)
target_compile_features(detect PUBLIC cxx_std_11)

unset(opencv_modules)
list(APPEND opencv_modules
    opencv_core
    opencv_calib3d
    opencv_dnn
    opencv_highgui
    opencv_ml
    opencv_objdetect
    opencv_photo
    opencv_shape
    opencv_stitching
    opencv_superres
    opencv_videostab
)
foreach(opencv_module ${opencv_modules})
    message(STATUS "Linking opencv module ${opencv_module} ...")
    target_link_libraries(detect
        ${opencv_module}
    )
    target_include_directories(detect
        PRIVATE $<TARGET_PROPERTY:${opencv_module},INCLUDE_DIRECTORIES>
    )
endforeach()
