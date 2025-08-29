# ===================================================================
# ChronoLattice | Root CMakeLists.txt
#
# This is the main project file that orchestrates the entire build.
# It sets global settings and includes the source directories.
# ===================================================================
cmake_minimum_required(VERSION 3.10)
project(ChronoLattice VERSION 0.0.1 LANGUAGES CXX)

# --- Global Project Settings ---
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Subdirectories ---
add_subdirectory(src)