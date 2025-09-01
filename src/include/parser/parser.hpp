#ifndef PARSER_PARSER_HPP
#define PARSER_PARSER_HPP

#include "core/schedule.hpp"

#include <filesystem>

namespace ChronoLattice {

/**
 * @brief Reads a task file and parses its contents into a ProjectGraph.
 *
 * @param filename The path to the task file.
 * @return A ProjectGraph representing the tasks and their dependencies.
 */
ProjectGraph parse_task_file(const std::filesystem::path& filename);

} // namespace ChronoLattice

#endif // PARSER_PARSER_HPP