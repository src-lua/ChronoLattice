#ifndef CORE_SCHEDULE_HPP_
#define CORE_SCHEDULE_HPP_

#include <string>
#include <vector>
#include <map>
#include <optional>

namespace ChronoLattice {

struct Task {
    std::string id;
    int duration;
    std::vector<std::string> predecessor_ids;
    std::string resource_id;
    int resource_qty;
};

using ProjectGraph = std::map<std::string, Task>;
using TaskOrdering = std::vector<std::string>;

class Schedule {
public:
    Schedule(const ProjectGraph& graph, TaskOrdering& ordering);
    Schedule(const Schedule& other);

    bool calculate();

    std::optional<int> getMakespan() const;
    
    void print() const;

private:
    const ProjectGraph& graph_;
    TaskOrdering& ordering_;
    std::map<std::string, int> start_times_;
    std::map<std::string, int> finish_times_;
    std::optional<int> makespan_;
};

} // namespace ChronoLattice

#endif // CORE_SCHEDULE_HPP_
