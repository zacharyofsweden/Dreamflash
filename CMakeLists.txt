cmake_minimum_required(VERSION 3.16)
project(Dreamflash LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(DREAMFLASH_SOURCES
    src/expert_io.c
    src/memory_manager.c
)

add_library(dreamflash_core STATIC ${DREAMFLASH_SOURCES})

if(WIN32)
    message(STATUS "Building Dreamflash Core for Windows (Win32 Overlapped IO)")
else()
    message(STATUS "Building Dreamflash Core for Linux / WSL2 (POSIX / io_uring)")
endif()
