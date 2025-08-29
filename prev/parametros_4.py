import ast
import math
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict

# Tipo de dado para uma tarefa, mantido para clareza
ResTask = Tuple[str, int, str, str, str, int]

class Project:
    """Representa um único cronograma de projeto e calcula suas métricas."""

    def __init__(self, id_cr: str, tasks_list: List[ResTask]):
        """Inicializa um projeto com sua ID e lista de tarefas."""
        self.id_cr = id_cr
        self._tasks_list = tasks_list
        self.task_ids = {t[0] for t in tasks_list}
        self.durations = {t[0]: t[1] for t in tasks_list}
        self._cpm_data = None  # Cache para resultados do CPM

    # --- Métodos para Cálculo do Caminho Crítico (CPM) ---

    def _topological_sort(self) -> Tuple[List[str], Dict, Dict]:
        """Realiza a ordenação topológica das tarefas (algoritmo de Kahn)."""
        preds = defaultdict(list)
        succs = defaultdict(list)
        for tid, _, po, pa, _, _ in self._tasks_list:
            all_preds = [p.strip() for p in (po.split(',') if po else []) +
                         (pa.split(',') if pa else []) if p.strip()]
            for p_id in all_preds:
                preds[tid].append(p_id)
                succs[p_id].append(tid)

        in_degree = {tid: len(preds[tid]) for tid in self.task_ids}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        topo_order = []
        while queue:
            u = queue.pop(0)
            topo_order.append(u)
            for v_id in succs[u]:
                in_degree[v_id] -= 1
                if in_degree[v_id] == 0:
                    queue.append(v_id)

        if len(topo_order) < len(self.task_ids):
            raise ValueError("Ciclo detectado no grafo de tarefas.")
        return topo_order, preds, succs

    def _calculate_early_times(self, topo_order, preds):
        """Calcula os tempos de início (ES) e fim (EF) mais cedo."""
        ES = {tid: 0 for tid in self.task_ids}
        EF = {}
        for tid in topo_order:
            ES[tid] = max((EF.get(p, 0) for p in preds[tid]), default=0)
            EF[tid] = ES[tid] + self.durations[tid]
        project_duration = max(EF.values(), default=0)
        return ES, EF, project_duration

    def _calculate_late_times(self, topo_order, succs, project_duration):
        """Calcula os tempos de início (LS) e fim (LFT) mais tarde."""
        LFT = {tid: project_duration for tid in self.task_ids}
        LS = {}
        for tid in reversed(topo_order):
            LFT[tid] = min((LS.get(s, project_duration) for s in succs[tid]),
                           default=project_duration)
            LS[tid] = LFT[tid] - self.durations[tid]
        return LS

    def calculate_cpm(self):
        """Orquestra o cálculo completo do CPM e armazena em cache."""
        if self._cpm_data is not None:
            return self._cpm_data

        try:
            topo, preds, succs = self._topological_sort()
            ES, EF, proj_dur = self._calculate_early_times(topo, preds)
            LS = self._calculate_late_times(topo, succs, proj_dur)

            crit_path = [tid for tid in topo if LS[tid] - ES[tid] == 0]
            slack = sum(LS[tid] - ES[tid] for tid in self.task_ids)

            self._cpm_data = {
                "critical_path": crit_path, "project_duration": proj_dur,
                "ES": ES, "EF": EF, "LS": LS, "slack": slack
            }
        except ValueError as e:
            # Marca como calculado mas falho para não tentar de novo
            self._cpm_data = {}
            raise e  # Propaga o erro para ser tratado pelo processador
        return self._cpm_data

    # --- Métodos para Cálculo de Métricas ---

    def get_cpm_summary_metrics(self) -> Dict:
        """Calcula e retorna as métricas de duração, folga, linearidade e expansão."""
        try:
            cpm_data = self.calculate_cpm()
        except ValueError:
            cpm_data = {}

        if not cpm_data:
            return {"duration": 0, "slack": 0, "linearity": 0.0, "expansion": 0.0}

        n = len(self._tasks_list)
        qtd_crit = len(cpm_data["critical_path"])

        q = 1.0
        if 1 < qtd_crit < n:
            q = 1 - math.log(qtd_crit) / math.log(n)

        exp = 0.0
        if qtd_crit > 0:
            crit_tasks_pa = sum(1 for tid, _, _, pa, _, _ in self._tasks_list
                                if tid in cpm_data["critical_path"] and pa)
            exp = crit_tasks_pa / qtd_crit

        return {"duration": cpm_data["project_duration"], "slack": cpm_data["slack"],
                "linearity": q, "expansion": exp}

    def calculate_regularity(self) -> float:
        """Calcula a métrica de regularidade da distribuição do trabalho."""
        try:
            cpm_data = self.calculate_cpm()
        except ValueError:
            return 0.0

        if not cpm_data or cpm_data["project_duration"] == 0:
            return 1.0

        max_day = cpm_data["project_duration"]
        daily_workload = np.zeros(max_day, dtype=int)
        for tid in self.task_ids:
            start, end = cpm_data["ES"].get(tid, 0), cpm_data["EF"].get(tid, 0)
            if start < end:
                daily_workload[start:end] += 1

        if daily_workload.sum() == 0:
            return 1.0

        cum_work = np.cumsum(np.concatenate(([0], daily_workload)))
        norm_curve = cum_work / cum_work[-1]
        ideal_curve = np.linspace(0, 1, len(norm_curve))

        abs_diff = np.abs(norm_curve - ideal_curve)
        dist_max = np.maximum(norm_curve, ideal_curve)

        return 1.0 - abs_diff.sum() / dist_max.sum() if dist_max.sum() > 0 else 0.0

    def _get_resource_histogram(self) -> Dict[str, List[int]]:
        """Gera um histograma de uso diário para cada recurso."""
        cpm_data = self.calculate_cpm()
        if not cpm_data: return {}

        max_day = cpm_data["project_duration"]
        resources = sorted({t[4] for t in self._tasks_list if t[4]})
        hist_by_res = {res: [0] * max_day for res in resources}

        for tid, _, _, _, r, qtd in self._tasks_list:
            if not r: continue
            start, end = cpm_data["ES"].get(tid, 0), cpm_data["EF"].get(tid, 0)
            for day in range(start, end):
                if day < max_day: hist_by_res[r][day] += qtd
        return hist_by_res

    def _get_filled_profile(self, profile: List[int]) -> List[int]:
        """Aplica o algoritmo "preenchimento de vales" a um perfil de uso."""
        n = len(profile)
        if n == 0: return []

        left_maxes = np.maximum.accumulate(profile)
        right_maxes = np.maximum.accumulate(profile[::-1])[::-1]

        return [min(left_maxes[i], right_maxes[i]) for i in range(n)]

    def calculate_productivity_metrics(self, all_resources: List[str]) -> Tuple[List[int], List[float]]:
        """Calcula o uso máximo e a produtividade para a lista de todos os recursos."""
        try:
            hist_by_res = self._get_resource_histogram()
        except ValueError:
            hist_by_res = {}

        max_vals, prod_vals = [], []
        for r in all_resources:
            profile = hist_by_res.get(r)
            if not profile or sum(profile) == 0:
                max_vals.append(0)
                prod_vals.append(1.0)
                continue

            filled_profile = self._get_filled_profile(profile)
            total_filled_work = sum(filled_profile)
            ratio = (sum(profile) / total_filled_work if total_filled_work > 0 else 0.0)
            max_vals.append(max(profile))
            prod_vals.append(ratio)
        return max_vals, prod_vals

class ScheduleProcessor:
    """Carrega, processa e salva os dados de múltiplos cronogramas."""

    def __init__(self, schedule_file: str, resource_file: str, output_file: str):
        """Inicializa o processador com os caminhos dos arquivos."""
        self.schedule_file = schedule_file
        self.resource_file = resource_file
        self.output_file = output_file
        self._resource_map: Dict[str, Tuple[str, int]] = {}
        self.all_resources: List[str] = []

    def _load_resources(self):
        """Carrega e processa o arquivo de tarefas e recursos, ignorando comentários no início."""
        try:
            with open(self.resource_file, 'r', encoding='utf-8') as f:
                full_content = f.read()
        except FileNotFoundError:
            print(f"ERRO CRÍTICO: Arquivo de recursos não encontrado em '{self.resource_file}'")
            raise

        # Encontra o início da lista de tarefas, ignorando quaisquer linhas de comentário.
        list_start_index = full_content.find('[')
        if list_start_index == -1:
            raise SyntaxError(f"Não foi possível encontrar o início da lista '[' no arquivo '{self.resource_file}'.")

        # Extrai e analisa apenas a parte do texto que contém a lista.
        list_content = full_content[list_start_index:]

        try:
            res_list = ast.literal_eval(list_content)
        except SyntaxError as e:
            print(f"ERRO CRÍTICO: Falha ao analisar a sintaxe do arquivo de recursos '{self.resource_file}'.")
            print(f"Detalhe do erro: {e}")
            raise

        # A verificação de 6 elementos por tupla continua sendo uma boa prática.
        for i, item in enumerate(res_list):
            if not isinstance(item, tuple) or len(item) != 6:
                raise ValueError(
                    f"Erro de formato em '{self.resource_file}', item {i+1}. "
                    f"Esperava tuplas com 6 elementos, mas o formato é inválido. "
                    f"Item com problema: {item}"
                )

        self._resource_map = {tid: (res, qtd) for tid, _, _, _, res, qtd in res_list if tid}
        self.all_resources = sorted({res for _, _, _, _, res, _ in res_list if res})

    def _generate_header(self) -> str:
        """Cria a string de cabeçalho para o arquivo CSV de saída."""
        if not self.all_resources:
            raise ValueError("Recursos devem ser carregados antes de gerar o cabeçalho.")
        
        header_parts = ["id", "duracao", "folga", "linearidade", "expansao", "regularidade"]
        max_header = [f"{r}_max" for r in self.all_resources]
        prod_header = [f"{r}_%" for r in self.all_resources]
        return ",".join(header_parts + max_header + prod_header) + "\n"

    def _process_line(self, line: str) -> str:
        """Processa uma única linha do arquivo de cronogramas e retorna a linha de resultado."""
        raw_data = ast.literal_eval(line.strip())
        id_cr = raw_data[0]
        
        tasks: List[ResTask] = []
        sched_list = [item for item in raw_data[1:] if isinstance(item, tuple) and len(item) == 4]
        for tid, dur, po, pa in sched_list:
            res, qtd = self._resource_map.get(tid, ('', 0))
            tasks.append((tid, dur, po, pa, res, qtd))

        project = Project(id_cr, tasks)
        
        cpm_metrics = project.get_cpm_summary_metrics()
        regularity = project.calculate_regularity()
        max_vals, prod_vals = project.calculate_productivity_metrics(self.all_resources)

        fields = [str(project.id_cr), str(cpm_metrics["duration"]), str(cpm_metrics["slack"]),
                  f"{cpm_metrics['linearity']:.4f}", f"{cpm_metrics['expansion']:.4f}",
                  f"{regularity:.4f}"]
        max_strs = [str(v) for v in max_vals]
        prod_strs = [f"{v:.2f}" for v in prod_vals]
        
        return ",".join(fields + max_strs + prod_strs) + "\n"

    def run(self):
        """Executa o processo completo: carrega, processa e salva."""
        print("Iniciando processamento...")
        self._load_resources()
        header = self._generate_header()
        
        with open(self.output_file, 'w', encoding='utf-8') as fout:
            fout.write(header)
            with open(self.schedule_file, 'r', encoding='utf-8') as fin:
                for i, line in enumerate(fin, 1):
                    try:
                        result_line = self._process_line(line)
                        fout.write(result_line)
                    except (ValueError, IndexError) as e:
                        print(f"Aviso: Cronograma na linha {i} ignorado devido a erro: {e}")
        
        print(f"Processamento concluído. Parâmetros salvos em: {self.output_file}")

def main():
    """Função principal para configurar e executar o processador."""
    # ATENÇÃO: Verifique se os caminhos dos arquivos estão corretos para o seu ambiente.
    base_path = r"C:\Users\farmc\Downloads\res_automacao"
    schedule_input_file = f"{base_path}\\R38_tarefas_4_resultados_unicos.txt"
    resource_input_file = f"{base_path}\\R38_tarefas_4.txt"
    output_param_file = f"{base_path}\\R38_tarefas_4_parametros.txt"
    
    processor = ScheduleProcessor(
        schedule_file=schedule_input_file,
        resource_file=resource_input_file,
        output_file=output_param_file
    )
    processor.run()

if __name__ == '__main__':
    main()