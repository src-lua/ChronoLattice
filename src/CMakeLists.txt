# ===================================================================
# ChronoLattice | Source CMakeLists.txt
#
# This CMakeLists.txt file is responsible for configuring the build.
# ===================================================================

message(STATUS "Configuring ChronoLattice (v${PROJECT_VERSION}).")

file(GLOB CRLT_SOURCES
    "src/engine/*.cpp"
    "src/parser/*.cpp"
    "src/main.cpp"
)

# --- Target Executable ---
add_executable(${PROJECT_NAME} ${CRLT_SOURCES})

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PUBLIC "include")

message(STATUS "Target '${PROJECT_NAME}' configured with ${CRLT_SOURCES}.")