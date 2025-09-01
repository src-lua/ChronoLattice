#include "core/schedule.hpp"
#include <vector>
#include <deque>
#include <iostream>

namespace ChronoLattice {

Schedule::Schedule(const ProjectGraph& graph, TaskOrdering& ordering) 
    : graph_(graph), ordering_(ordering), makespan_(std::nullopt) {}

Schedule::Schedule(const Schedule& other) 
    : graph_(other.graph_),
      ordering_(other.ordering_),
      start_times_(other.start_times_), 
      finish_times_(other.finish_times_), 
      makespan_(other.makespan_) {}

bool Schedule::calculate() {
    start_times_.clear();
    finish_times_.clear();
    makespan_ = std::nullopt;

    if (ordering_.empty()) return false;

    for (const std::string& task_id : ordering_) {
        int earliest_start = 0;
        const auto& task = graph_.at(task_id);
        for (const std::string& pred_id : task.predecessor_ids) {
            if (finish_times_.count(pred_id)) {
                earliest_start = std::max(earliest_start, finish_times_.at(pred_id));
            }
        }
        
        start_times_[task_id] = earliest_start;
        finish_times_[task_id] = earliest_start + task.duration;
    }

    int max_finish_time = 0;
    for (const auto& pair : finish_times_) {
        max_finish_time = std::max(max_finish_time, pair.second);
    }
    makespan_ = max_finish_time;

    return true;
}

std::optional<int> Schedule::getMakespan() const {
    return makespan_;
}

void Schedule::print() const {
    if (!makespan_) {
        std::cout << "Cronograma nao calculado ou invalido." << std::endl;
        return;
    }
    std::cout << "\n--- Detalhes do Cronograma ---" << std::endl;
    std::cout << "Duracao Total (Makespan): " << *makespan_ << " dias" << std::endl;

    // Print the five first for example
    for(int i = 1; i <= 5 && i <= graph_.size(); ++i) {
        std::string id = std::to_string(i);
        if(start_times_.count(id)) {
            std::cout << "  - Tarefa " << id 
                      << ": Inicio = " << start_times_.at(id) 
                      << ", Fim = " << finish_times_.at(id) << std::endl;
        }
    }
}

} // namespace ChronoLattice