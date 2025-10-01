cmake_minimum_required(VERSION 4.0.2 FATAL_ERROR)
cmake_policy(SET CMP0167 NEW) # new find boost
# Dependencies; Boost:
find_package(Boost 1.81.0 REQUIRED COMPONENTS headers)

option(ENABLE_LLD "Use LLD? Off by default." OFF)
option(USE_MI_MALLOC "Use MI_MALLOC if available." ON)
option(USE_JE_MALLOC "Use jemallc if available and not using mi_malloc." ON)
option(ENABLE_WIDE_VECTORS "Compile with 512bit vectors if available" ON)
option(POLYMATHNOEXPLICITSIMDARRAY "No explicit SIMD for Array operations" OFF)
# option(FOR_TESTING "Build for testing" OFF)

add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions;-fno-rtti>")
set(CMAKE_EXPERIMENTAL_CXX_IMPORT_STD
    # This specific value changes as experimental support evolves. See
    # `Help/dev/experimental.rst` in the CMake source corresponding to
    # your CMake build for the exact value to use.
    "d0edc3af-4c50-42ea-a356-e2862fe7a444")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(
  Matmul
  VERSION 1.0
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_MODULE_STD ON)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

file(GLOB_RECURSE headers CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hxx")
file(GLOB_RECURSE module_interfaces CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/mod/*.cxxm")
file(GLOB_RECURSE module_implementations CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/mod/*.cxx")

message(STATUS "module_interfaces = ${module_interfaces}")
message(STATUS "module_implementations = ${module_implementations}")

# ---- Create library ----
add_library(${PROJECT_NAME} OBJECT) # ${headers})
target_sources(
  ${PROJECT_NAME}
  PUBLIC FILE_SET CXX_MODULES FILES ${module_interfaces}
  PRIVATE ${module_implementations}
  INTERFACE FILE_SET HEADERS FILES ${headers})
target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>)
# --- Maybe Use SIMD Array Ops ---
if(POLYMATHNOEXPLICITSIMDARRAY)
  target_compile_definitions(${PROJECT_NAME}
                             PRIVATE POLYMATHNOEXPLICITSIMDARRAY)
endif()
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_23)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(
    ${PROJECT_NAME} PRIVATE -ferror-limit=8 -fcolor-diagnostics -ftime-trace)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(
    ${PROJECT_NAME}
    PRIVATE -fmax-errors=8 -fconcepts-diagnostics-depth=4
            -fno-semantic-interposition -fdiagnostics-color=always
            -Wno-comma-subscript -Wno-psabi)
endif()
target_compile_options(${PROJECT_NAME} PRIVATE -Wshadow -Werror) # -Wpedantic
                                                                 # -Wextra -Wall
# -Wno-reference-tu-local-entity-in-other-tu)
target_compile_features(
  ${PROJECT_NAME}
  PRIVATE cxx_std_23
  INTERFACE cxx_std_23)
set_target_properties(
  ${PROJECT_NAME}
  PROPERTIES CXX_STANDARD 23
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN ON)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 23)

# being a cross-platform target, we enforce standards conformance on MSVC
# target_compile_options( ${PROJECT_NAME} INTERFACE
# "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# --- Allocator override ----
set(SEARCH_JE_MALLOC USE_JE_MALLOC)
if(USE_MI_MALLOC)
  find_package(mimalloc) # REQUIRED)
  if(mimalloc_FOUND)
    target_link_libraries(${PROJECT_NAME} PRIVATE mimalloc) # mimalloc-static
    target_compile_definitions(${PROJECT_NAME} PRIVATE USING_MIMALLOC)
    set(SEARCH_JE_MALLOC OFF)
  endif()
endif()
if(SEARCH_JE_MALLOC)
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(JEMALLOC jemalloc)
    if(jemalloc_FOUND)
      target_link_libraries(${PROJECT_NAME} PRIVATE ${JEMALLOC_LIBRARIES})
      target_include_directories(${PROJECT_NAME}
                                 PRIVATE ${JEMALLOC_INCLUDE_DIRS})
      target_compile_definitions(${PROJECT_NAME} PRIVATE USING_JEMALLOC)
    endif()
  endif()
endif()

set(BOOST_UT_DISABLE_MODULE OFF)
add_subdirectory("${CMAKE_SOURCE_DIR}/extern/ut" "extern_build/ut")
file(GLOB test_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp)

# Create a test executable for each source file
foreach(test_source ${test_sources})
  get_filename_component(test_name ${test_source} NAME_WE)
  add_executable(${test_name} ${test_source})
  target_link_libraries(${test_name} PRIVATE Matmul ut_module)
  set_target_properties(
    ${test_name}
    PROPERTIES CXX_STANDARD 23
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON)

  # Apply the same compiler options as the main target
  if((USE_SANITIZER MATCHES "([Uu]ndefined)")
     OR (USE_SANITIZER MATCHES "([Aa]ddress);([Uu]ndefined)"))
    target_compile_options(${test_name} PRIVATE -fsanitize-trap=all)
  elseif(USE_TYPE_SANITIZER)
    target_compile_options(${test_name} PRIVATE -fsanitize-trap=all
                                                -fsanitize=type)
    target_link_options(${test_name} PRIVATE -fsanitize=type)
  endif()

  if(POLYMATHNOEXPLICITSIMDARRAY)
    target_compile_definitions(${test_name} PRIVATE POLYMATHNOEXPLICITSIMDARRAY)
  endif()

  target_compile_options(${test_name} PUBLIC -Wall -Wpedantic -Wextra -Wshadow
                                             -Werror)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(
      ${test_name} PRIVATE -ferror-limit=8 -fcolor-diagnostics -ftime-trace)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(
      ${test_name}
      PRIVATE -fmax-errors=8 -fconcepts-diagnostics-depth=4
              -fno-semantic-interposition -fdiagnostics-color=always
              -Wno-comma-subscript -Wno-psabi)
  endif()

  target_compile_options(${test_name} PRIVATE -D_GLIBCXX_ASSERTIONS
                                              -ftemplate-backtrace-limit=0)

  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Only fetch nanobench if we have benchmarks to build
# ---- Add Benchmarks ----
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  file(GLOB benchmark_sources CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp)
  if(benchmark_sources)
    include(FetchContent)
    FetchContent_Declare(
      nanobench
      GIT_REPOSITORY https://github.com/Spartan322/nanobench
      GIT_TAG f49b052a68bc05d5497acf1f6c0c2d6ef65c6eef
      GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nanobench)

    file(GLOB benchmark_modules CONFIGURE_DEPENDS
         ${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cxxm)

    # Create a benchmark executable for each source file
    foreach(benchmark_source ${benchmark_sources})
      get_filename_component(benchmark_name ${benchmark_source} NAME_WE)
      add_executable(${benchmark_name} ${benchmark_source})
      target_sources(${benchmark_name} PUBLIC FILE_SET CXX_MODULES FILES
                                              ${benchmark_modules})
      target_link_libraries(${benchmark_name} PRIVATE Matmul
                                                      nanobench::nanobench)
      set_target_properties(
        ${benchmark_name}
        PROPERTIES CXX_STANDARD 23
                   CXX_VISIBILITY_PRESET hidden
                   VISIBILITY_INLINES_HIDDEN ON)

      # Optimization flags for benchmarks
      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(
          ${benchmark_name} PRIVATE -O3 -march=native -DNDEBUG -ferror-limit=8
                                    -fcolor-diagnostics)
      elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(
          ${benchmark_name}
          PRIVATE -O3
                  -march=native
                  -DNDEBUG
                  -fmax-errors=8
                  -fconcepts-diagnostics-depth=4
                  -fno-semantic-interposition
                  -fdiagnostics-color=always
                  -Wno-comma-subscript
                  -Wno-psabi)
      endif()

      target_compile_options(${benchmark_name} PRIVATE -Wshadow -Werror)

      if(POLYMATHNOEXPLICITSIMDARRAY)
        target_compile_definitions(${benchmark_name}
                                   PRIVATE POLYMATHNOEXPLICITSIMDARRAY)
      endif()
    endforeach()
  endif()
endif()

# ---- Add MathTests ----

include(CTest)
enable_testing()
