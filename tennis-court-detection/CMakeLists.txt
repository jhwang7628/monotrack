cmake_minimum_required(VERSION 3.16)
project(detect)

# Fetch opencv from source
set(FETCHCONTENT_BASE_DIR "${CMAKE_SOURCE_DIR}/external" CACHE PATH "Directory under which to collect all populated content")

list(PREPEND CMAKE_MODULE_PATH
    "${PROJECT_SOURCE_DIR}/cmake/recipes/external/"
    "${PROJECT_SOURCE_DIR}/cmake/recipes/internal/"
)

include(opencv)

file(GLOB INC_FILES "*.h")
file(GLOB SRC_FILES "*.cpp")
add_executable(detect ${INC_FILES} ${SRC_FILES})

message("Jui: opencv_core = ${OpenCVModules_TARGETS}")

target_include_directories(detect PRIVATE
    /home/juiwang/code/ai-badminton/tennis-court-detection/external/opencv-src/include
    #/home/juiwang/code/ai-badminton/tennis-court-detection/external/opencv-src/modules/core/include
)
get_target_property(detect_inc detect INCLUDE_DIRECTORIES)
message("Jui: detect_in is = ${detect_inc}")
if (TARGET opencv_core)
    message("Jui: opencv_core is a target")
    get_target_property(opencv_core_include_directory opencv_core INCLUDE_DIRECTORIES)
    message("Jui: opencv_core has include directories ${opencv_core_include_directory}")
else()
    message("Jui: opencv_core is NOT a target")
endif()

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


#target_link_libraries(detect opencv_core)
#target_include_directories(detect
#    PRIVATE $<TARGET_PROPERTY:opencv_core,INCLUDE_DIRECTORIES>
#    PRIVATE $<TARGET_PROPERTY:opencv_calib3d,INCLUDE_DIRECTORIES>
#)
get_target_property(detect_inc detect INCLUDE_DIRECTORIES)
message("Jui: detect_in is now = ${detect_inc}")
message("Jui: opencv core headers ${OPENCV_MODULE_opencv_core_HEADERS}")

#set(OpenCV_DIR ${CMAKE_CURRENT_BINARY_DIR})
#find_package(OpenCV REQUIRED)
#message("Jui: OpenCV_INCLUDE_DIR = ${OpenCV_INCLUDE_DIR}")
#target_compile_features(detect PUBLIC cxx_std_11)
#target_include_directories(detect PRIVATE
#    ${OpenCV_INCLUDE_DIR})
#target_link_libraries(detect PRIVATE
#    ${OpenCV_LIBS})
