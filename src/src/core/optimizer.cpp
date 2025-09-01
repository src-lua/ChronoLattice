#include "core/optimizer.hpp"
#include <cmath>
#include <iostream>
#include <queue>
#include <random>

namespace ChronoLattice {    

Optimizer::Optimizer(const ProjectGraph& graph) 
    : graph_(graph)
{
    current_ordering_ = generateInitialOrdering();
    best_ordering_ = current_ordering_;
    best_cost_ = calculateCost(best_ordering_);
}

double Optimizer::calculateCost(TaskOrdering& ordering) {
    Schedule schedule(graph_, ordering);
    if (schedule.calculate()) {
        auto makespan = schedule.getMakespan();
        if (makespan) {
            return static_cast<double>(*makespan);
        }
    }
    return std::numeric_limits<double>::infinity();
}

TaskOrdering Optimizer::generateInitialOrdering() {
    TaskOrdering order;
    std::map<std::string, int> in_degree;
    std::map<std::string, std::vector<std::string>> successors;

    for (const auto& [u, task] : graph_) in_degree[u] = 0;

    for (const auto& [u, task] : graph_) {
        for (const auto& pred_id : task.predecessor_ids) {
            if (graph_.count(pred_id)) {
                successors[pred_id].push_back(u);
                in_degree[u]++;
            }
        }
    }

    // Topological sort with Kahn's algorithm
    std::deque<std::string> queue;
    for (const auto& pair : in_degree) {
        if (pair.second == 0) queue.push_back(pair.first);
    }

    while (!queue.empty()) {
        std::string u = queue.front();
        queue.pop_front();
        order.push_back(u);

        for (const auto& v : successors[u]) {
            in_degree[v]--;
            if (in_degree[v] == 0) queue.push_back(v);
        }
    }
    return order;
}

TaskOrdering Optimizer::generateNeighbor(TaskOrdering ordering) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    // Not enough neighbors to swap
    if (ordering.size() < 2) return ordering;

    std::uniform_int_distribution<> distrib(0, ordering.size() - 2);

    // TODO: Improve swap logic
    // Tries up to 10 times to find a valid swap
    for(int i = 0; i < 10; ++i) {
        int idx1 = distrib(gen);
        int idx2 = idx1 + 1;

        const auto& task1_id = ordering[idx1];
        const auto& task2_id = ordering[idx2];
        const auto& task2_preds = graph_.at(task2_id).predecessor_ids;

        // task1 CANNOT be a predecessor of task2
        bool is_dependent = false;
        for(const auto& pred_id : task2_preds) {
            if (pred_id == task1_id) {
                is_dependent = true;
                break;
            }
        }

        if (!is_dependent) {
            std::swap(ordering[idx1], ordering[idx2]);
            return ordering;
        }
    }

    std::cerr << "Nao foi possivel encontrar uma troca valida." << std::endl;
    return ordering;
}

Schedule Optimizer::run() {
    std::cout << "\nIniciando otimizacao com Simulated Annealing..." << std::endl;
    std::cout << "Custo Inicial (Makespan): " << best_cost_ << std::endl;

    double current_cost = best_cost_;

    // --- Simulated Annealing parameters ---
    double temperature = 1000.0;
    double cooling_rate = 0.995;
    int max_iterations = 2000;

    // --- Random number generator ---
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    // --- Simulated Annealing loop ---
    for (int i = 0; i < max_iterations && temperature > 1.0; ++i) {
        TaskOrdering neighbor_ordering = generateNeighbor(current_ordering_);
        double neighbor_cost = calculateCost(neighbor_ordering);

        if (neighbor_cost < current_cost) { // Better Solution! Always accept.
            current_ordering_ = neighbor_ordering;
            current_cost = neighbor_cost;
        } 
        else { // Worse Solution. Maybe accept.
            // Euler comes from Boltzmann distribution
            double acceptance_prob = std::exp((current_cost - neighbor_cost) / temperature);
            if (distrib(gen) < acceptance_prob) {
                current_ordering_ = neighbor_ordering;
                current_cost = neighbor_cost;
            }
        }

        // Update best solution found so far
        if (current_cost < best_cost_) {
            best_ordering_ = current_ordering_;
            best_cost_ = current_cost;
        }

        temperature *= cooling_rate;
    }

    std::cout << "Otimizacao concluida." << std::endl;
    std::cout << "Melhor Custo Encontrado (Makespan): " << best_cost_ << std::endl;

    // Return final schedule based on best ordering found
    Schedule final_schedule(graph_, best_ordering_);
    final_schedule.calculate();
    return final_schedule;
}

} // namespace ChronoLattice