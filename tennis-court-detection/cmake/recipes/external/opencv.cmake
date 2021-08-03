if (TARGET opencv)
    return()
endif()

message(STATUS "Third-party (external): creating target 'opencv'...")

include(FetchContent)
FetchContent_Declare(
    opencv
    GIT_REPOSITORY https://github.com/opencv/opencv.git
    GIT_TAG        3.4.12
    )

# opencv options
option(WITH_FFMPEG "Include FFMPEG support" ON)

FetchContent_MakeAvailable(opencv)
