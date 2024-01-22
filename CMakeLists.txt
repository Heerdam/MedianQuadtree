
cmake_minimum_required(VERSION 3.3)

project(Test)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS_DEBUG)
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -ggdb" )
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG -funroll-loops")
endif()

include(FetchContent)

add_executable(mqt
   src/main.cpp
)

# ------------------ LIBMORTON ------------------
# message(STATUS "Fetching libmorton...")
# FetchContent_Declare(
#     libmorton 
#    GIT_REPOSITORY      https://github.com/Forceflow/libmorton.git
#     GIT_TAG             v0.2.12
# )
# set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(libmorton)

# ------------------ ROBIN-MAP ------------------
# message(STATUS "Fetching robin map...")
# FetchContent_Declare(robin_map
#   GIT_REPOSITORY        https://github.com/Tessil/robin-map.git
#   GIT_TAG               v1.2.1
# )
# FetchContent_MakeAvailable(robin_map)

target_include_directories(mqt PUBLIC include )

target_link_libraries(mqt
# 	libmorton
# 	tsl::robin_map
)
