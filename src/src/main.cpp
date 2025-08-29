#include "engine/ChronoLattice.hpp"
#include "parser/parser.hpp"


#include <iostream>
#include <fstream>
#include <sstream>
#include <filesystem>

int main() {
    std::cout << "Iniciando o Otimizador de Cronograma..." << std::endl;

    auto filename = std::filesystem::path("tarefas.txt");
    ChronoLattice::ProjectGraph graph = ChronoLattice::parse_task_file(filename);

    if (graph.empty()) {
        std::cerr
            << "Nenhuma tarefa foi carregada. Verifique o arquivo de entrada."
            << std::endl;
        return 1;
    }

    std::cout << "Arquivo lido com sucesso!" << std::endl;
    std::cout << "Total de tarefas carregadas: " << graph.size() << std::endl;

    if (graph.count("1")) {
        const auto& task1 = graph.at("1");
        std::cout << "\n--- Exemplo: Tarefa 1 ---" << std::endl;
        std::cout << "ID: " << task1.id << std::endl;
        std::cout << "Duracao: " << task1.duration << " dias" << std::endl;
        std::cout << "Recurso: " << task1.resource_id
                  << " (Qtd: " << task1.resource_qty << ")" << std::endl;
        std::cout << "Predecessores: ";
        for (const auto& pred : task1.predecessor_ids) {
            std::cout << pred << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}