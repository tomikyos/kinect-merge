cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(kinect_merge)

set(Boost_USE_STATIC_LIBS        ON)
set(Boost_USE_MULTITHREADED      ON)
set(Boost_USE_STATIC_RUNTIME    OFF)
find_package(Boost COMPONENTS system filesystem program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(OpenCV_DIR "../../OpenCV-2.3.1/build" CACHE PATH "Path to the OpenCV build directory")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB_RECURSE INCS "*.h")
set(SRCS 
    kinect_merge.cpp
    kinect_view.cpp
    kinect_calibration.cpp)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -DDEBUG")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -ggdb -DNDEBUG")
endif()

add_executable (kinect_merge ${SRCS} ${INCS})
target_link_libraries (kinect_merge
	${OpenCV_LIBS} 
	${Boost_LIBRARIES})
