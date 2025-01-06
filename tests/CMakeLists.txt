cmake_minimum_required(VERSION 3.29.2)

project(mqt2_tests)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.7.1
  GIT_SHALLOW True
)

FetchContent_MakeAvailable(Catch2)

add_executable(mqt2_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests.cpp)
target_link_libraries(mqt2_tests PRIVATE Catch2::Catch2WithMain)