cmake_minimum_required(VERSION 3.2)

set(CMAKE_CXX_STANDARD 17)

project(Message_Passing_NN 	VERSION 1.0)


INCLUDE_DIRECTORIES(eigen/)

add_custom_target(copy-runtime-files ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/resources ${CMAKE_BINARY_DIR}/resources
    DEPENDS ${MY_TARGET})

set(SOURCE_FILES main.cpp data.h layers.h GCN_Model.h)
add_executable(out ${SOURCE_FILES})

install(TARGETS out DESTINATION bin)
