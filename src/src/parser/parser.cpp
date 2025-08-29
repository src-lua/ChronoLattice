#include "parser/parser.hpp"

#include <sstream>
#include <fstream>
#include <algorithm>
#include <iostream>

namespace ChronoLattice {

std::vector<std::string> parse_predecessors(std::string pred_str) {
    std::vector<std::string> preds;
    pred_str.erase(std::remove(pred_str.begin(), pred_str.end(), '\''), pred_str.end());
    
    if (pred_str.empty()) {
        return preds;
    }

    std::stringstream ss(pred_str);
    std::string pred;
    while (std::getline(ss, pred, ',')) {
        if (!pred.empty()) {
            preds.push_back(pred);
        }
    }
    return preds;
}

ProjectGraph parse_task_file(const std::filesystem::path& filename) {
    ProjectGraph graph;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Erro: Nao foi possivel abrir o arquivo " << filename << std::endl;
        return graph;
    }

    std::string line;

    std::string content;
    while (std::getline(file, line)) {
        // Ignora comentários
        if (line.rfind("#", 0) == 0) continue;
        content += line;
    }

    size_t start_pos = content.find('[');
    size_t end_pos = content.find(']');
    if (start_pos == std::string::npos || end_pos == std::string::npos) {
        std::cerr << "Erro: Formato de arquivo invalido. Não encontrou '[' ou ']'." << std::endl;
        return graph;
    }

    // Pega apenas o conteúdo entre '[' e ']'
    content = content.substr(start_pos + 1, end_pos - start_pos - 1);

    std::stringstream ss(content);
    std::string task_str;

    // Divide o conteúdo em tuplas, separando por ')'
    while (std::getline(ss, task_str, ')')) {
        // Remove lixo inicial (vírgulas, parênteses, espaços)
        size_t first_char_pos = task_str.find_first_not_of(", \t\n\r(");
        if (first_char_pos == std::string::npos) continue; // Linha vazia ou lixo
        task_str = task_str.substr(first_char_pos);
        
        if (task_str.empty()) continue;

        std::vector<std::string> parts;
        std::stringstream task_ss(task_str);
        std::string part;
        
        while(std::getline(task_ss, part, ',')) {
            parts.push_back(part);
        }

        if (parts.size() > 6) {
            std::string combined_preds = parts[2];
            for (size_t i = 3; i < parts.size() - 3; ++i) {
                combined_preds += "," + parts[i];
            }
            std::vector<std::string> new_parts;
            new_parts.push_back(parts[0]);
            new_parts.push_back(parts[1]);
            new_parts.push_back(combined_preds);
            new_parts.push_back(parts[parts.size() - 3]);
            new_parts.push_back(parts[parts.size() - 2]);
            new_parts.push_back(parts[parts.size() - 1]);
            parts = new_parts;
        }

        if (parts.size() == 6) {
            try {
                Task t;
                
                for(auto& p : parts) {
                    p.erase(std::remove(p.begin(), p.end(), '\''), p.end());
                    p.erase(0, p.find_first_not_of(" \t"));
                    p.erase(p.find_last_not_of(" \t") + 1);
                }

                t.id = parts[0];
                t.duration = std::stoi(parts[1]);
                t.predecessor_ids = parse_predecessors(parts[2]);
                // parts[3] is ignored
                t.resource_id = parts[4];
                t.resource_qty = std::stoi(parts[5]);
                
                graph[t.id] = t;

            } catch (const std::invalid_argument& e) {
                std::cerr << "Erro de conversão em: " << task_str << " -> " << e.what() << std::endl;
            }
        }
    }

    return graph;
}

} // namespace ChronoLattice