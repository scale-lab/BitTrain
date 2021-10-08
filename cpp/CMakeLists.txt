cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(bitmap-tensor)

find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example 
               src/tbitmap.cpp
               src/example.cpp
)

target_include_directories(example
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        "${PYTHON_INCLUDE_DIRS}"
        /usr/local/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(example
                      "${TORCH_LIBRARIES}"
                      "${PYTHON_LIBRARIES}"                  
)

set_property(TARGET example PROPERTY CXX_STANDARD 14)

