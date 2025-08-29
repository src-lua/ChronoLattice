from typing import List, Any, Dict, Tuple, Union
import copy, ast

CronId = Union[str, int]

def network_to_arranjo(tasks: List[List[Any]]) -> List[List[Any]]:
    """
    Converte uma lista de tarefas em um arranjo diário (dia, tarefas).
    Cada tarefa: [nome, duração, predecessora1, predecessora2]
    """
    duracoes: Dict[str, int] = {}
    preds: Dict[str, List[str]] = {}
    for nome, duracao, p1, p2 in tasks:
        duracoes[nome] = duracao
        # Lida com predecessoras múltiplas (ex: 'T1,T2') em um único campo
        all_preds = []
        for pred_group in (p1, p2):
            if pred_group:
                all_preds.extend(p.strip() for p in pred_group.split(',') if p.strip())
        preds[nome] = all_preds

    arranjo: List[List[Any]] = []
    dia = 1
    while True:
        # .get(p, 0) evita erro se uma predecessora não estiver na lista de tarefas (ex: redes desconexas)
        ativas = [t for t, d in duracoes.items() if d > 0 and all(duracoes.get(p, 0) == 0 for p in preds.get(t, []))]
        if not ativas: break
        arranjo.append([dia] + sorted(ativas))
        for t in ativas:
            duracoes[t] -= 1
        dia += 1
    return arranjo

def prune_networks_by_arranjo(
    networks: List[Tuple[CronId, List[List[Any]]]]
) -> Tuple[
    List[Tuple[CronId, List[List[Any]]]],
    List[Tuple[CronId, List[List[Any]]]]
]:
    """
    Separa redes em únicas e duplicadas com base em arranjos equivalentes.
    Recebe lista de (cron_id, tasks).
    Retorna (únicas, duplicadas).
    """
    vistos = set()
    unicos: List[Tuple[CronId, List[List[Any]]]] = []
    duplicados: List[Tuple[CronId, List[List[Any]]]] = []
    for cron_id, tasks in networks:
        arr = network_to_arranjo(copy.deepcopy(tasks))
        key = str(arr)
        if key not in vistos:
            vistos.add(key)
            unicos.append((cron_id, tasks))
        else:
            duplicados.append((cron_id, tasks))
    return unicos, duplicados

def read_networks(input_path: str) -> List[Tuple[CronId, List[List[Any]]]]:
    """
    Lê redes de arquivo com linhas no formato:
    [cron_id, (id, duracao, pred1, pred2), ...]
    Return: lista de (cron_id, tasks) onde tasks é lista de [id, duracao, pred1, pred2]
    """
    networks: List[Tuple[CronId, List[List[Any]]]] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                parsed = ast.literal_eval(raw)
                # Deve ser lista com ao menos 2 itens: cron_id + tarefas
                if isinstance(parsed, (list, tuple)) and len(parsed) >= 2:
                    cron_id = parsed[0]
                    if not isinstance(cron_id, (int, str)):
                        continue
                    tasks: List[List[Any]] = []
                    valid = True
                    for item in parsed[1:]:
                        if isinstance(item, (list, tuple)) and len(item) == 4:
                            # item: (id, duracao, pred1, pred2)
                            tasks.append([item[0], item[1], item[2], item[3]])
                        else:
                            valid = False
                            break
                    if valid:
                        networks.append((cron_id, tasks))
            except (ValueError, SyntaxError):
                continue
    return networks

def write_networks(
    output_path: str,
    networks: List[Tuple[CronId, List[List[Any]]]]
) -> None:
    """
    Escreve redes no mesmo formato de entrada:
    [cron_id, (id, duracao, pred1, pred2), ...]
    Cada rede por linha.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for cron_id, tasks in networks:
            # Constrói lista para repr
            lista_repr = [cron_id] + [tuple(item) for item in tasks]
            f.write(repr(lista_repr) + "\n")


if __name__ == '__main__':
    input_file = r"C:\Users\farmc\Downloads\res_automacao\R27_tarefas_4_resultados_unicos.txt"
    podados_file = r"C:\Users\farmc\Downloads\res_automacao\podados.txt"
    duplicados_file = r"C:\Users\farmc\Downloads\res_automacao\duplicados.txt"

    redes = read_networks(input_file)
    redes_unicas, redes_duplicadas = prune_networks_by_arranjo(redes)
    write_networks(podados_file, redes_unicas)
    write_networks(duplicados_file, redes_duplicadas)

    print(f"Total de redes lidas: {len(redes)}")
    print(f"Redes únicas gravadas em: {podados_file} (total: {len(redes_unicas)})")
    print(f"Redes duplicadas gravadas em: {duplicados_file} (total: {len(redes_duplicadas)})")