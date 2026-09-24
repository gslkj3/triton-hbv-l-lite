cmake_minimum_required(VERSION 3.20)
project(LCorePasses LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)
set(LCORE_TRITON_SOURCE_DIR "" CACHE PATH "External pinned Triton source")
set(LCORE_TRITON_BINARY_DIR "" CACHE PATH "External Triton generated headers")
foreach(required
    "${LCORE_TRITON_SOURCE_DIR}/include/triton/Dialect/Triton/IR/Dialect.h"
    "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen")
  if(NOT EXISTS "${required}")
    message(FATAL_ERROR "Missing declared external compiler dependency: ${required}")
  endif()
endforeach()
if(NOT TARGET TritonIR AND NOT EXISTS "${LCORE_TRITON_BINARY_DIR}/include/triton/Dialect/Triton/IR/Ops.h.inc")
  message(FATAL_ERROR "External Triton generated IR headers are missing")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/LCore")
add_custom_command(
  OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/include/LCore/Passes.h.inc"
  COMMAND "${LLVM_TOOLS_BINARY_DIR}/mlir-tblgen" -gen-pass-decls
          "-I${LLVM_INCLUDE_DIRS}"
          "${CMAKE_CURRENT_SOURCE_DIR}/include/LCore/Passes.td"
          -o "${CMAKE_CURRENT_BINARY_DIR}/include/LCore/Passes.h.inc"
  DEPENDS include/LCore/Passes.td
          "${LLVM_INCLUDE_DIRS}/mlir/Pass/PassBase.td"
  VERBATIM)
add_library(LCorePasses STATIC lib/LoopPasses.cpp
  "${CMAKE_CURRENT_BINARY_DIR}/include/LCore/Passes.h.inc")
target_compile_features(LCorePasses PUBLIC cxx_std_17)
if(TARGET TritonIR)
  add_dependencies(LCorePasses TritonIR)
endif()
set_target_properties(LCorePasses PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(LCorePasses PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/include"
  "${LCORE_TRITON_SOURCE_DIR}/include" "${LCORE_TRITON_BINARY_DIR}/include"
  ${LLVM_INCLUDE_DIRS})
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(LCorePasses PRIVATE -fno-rtti)
endif()
if(NOT LLVM_ENABLE_EH)
  target_compile_options(LCorePasses PRIVATE -fno-exceptions)
endif()

# This archive is linked into a compiler that supplies native Triton/MLIR IR
# libraries. It is not a replacement installation of LLVM or Triton.
include(GNUInstallDirs)
install(TARGETS LCorePasses ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/LCore DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
