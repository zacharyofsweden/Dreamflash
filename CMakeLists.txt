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
    # io_uring is not implemented; the POSIX backend is synchronous pread.
    message(STATUS "Building Dreamflash Core for Linux / WSL2 (POSIX pread, synchronous)")
endif()

# The C engine had no caller of any kind until this target existed.
enable_testing()
add_executable(test_expert_io tests/test_expert_io.c)
target_link_libraries(test_expert_io PRIVATE dreamflash_core)
add_test(NAME expert_io COMMAND test_expert_io)
