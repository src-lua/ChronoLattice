#ifndef CORE_OPTIMIZER_HPP
#define CORE_OPTIMIZER_HPP

#include "schedule.hpp"

namespace ChronoLattice {

class Optimizer {
public:
    explicit Optimizer(const ProjectGraph& graph);
    Schedule run();

private:
    const ProjectGraph& graph_;

    // --- Optimization State ---
    TaskOrdering current_ordering_;
    TaskOrdering best_ordering_;
    double best_cost_;

    // --- Auxiliary Functions ---
    double calculateCost(TaskOrdering& ordering);
    TaskOrdering generateInitialOrdering();
    TaskOrdering generateNeighbor(TaskOrdering ordering);
};

} // namespace ChronoLattice

#endif // CORE_OPTIMIZER_HPP
