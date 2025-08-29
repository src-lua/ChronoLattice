#ifndef CHRONO_LATTICE_HPP_
#define CHRONO_LATTICE_HPP_

#include <string>
#include <vector>
#include <map>

namespace ChronoLattice {

struct Task {
    std::string id;
    int duration;
    std::vector<std::string> predecessor_ids;
    std::string resource_id;
    int resource_qty;
};

using ProjectGraph = std::map<std::string, Task>;


} // namespace ChronoLattice

#endif // CHRONO_LATTICE_HPP_
