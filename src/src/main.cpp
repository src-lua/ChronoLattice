#include <filesystem>
#include <iostream>

#include "core/optimizer.hpp"
#include "parser/parser.hpp"

int main() {    
    std::cout << "Iniciando o Otimizador de Cronograma..." << std::endl;

    auto filename = std::filesystem::path("tarefas.txt");
    ChronoLattice::ProjectGraph graph =
        ChronoLattice::parse_task_file(filename);

    if (graph.empty()) {
        std::cerr
            << "Nenhuma tarefa foi carregada. Verifique o arquivo de entrada."
            << std::endl;
        return 1;
    }
    std::cout << "Total de tarefas carregadas: " << graph.size() << std::endl;

    ChronoLattice::Optimizer optimizer(graph);

    ChronoLattice::Schedule best_schedule_found = optimizer.run();

    std::cout << "\n--- Melhor Cronograma Encontrado (Apos Otimizacao) ---";
    best_schedule_found.print();

    return 0;
}