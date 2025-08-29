# -*- coding: utf-8 -*-

import time, itertools, ast, multiprocessing, os, math, sys, logging, logging.handlers, hashlib, psutil, threading, traceback
from collections import deque, defaultdict, OrderedDict
from typing import Any, List, Tuple, Dict, Optional, Set, Callable
from functools import lru_cache
from itertools import product

class MemUtil:
    _enabled = False
    _have_psutil = False
    _max_rss_mb = 0.0
    _sample_period = 3.0
    _stop = False
    _bg_thread = None

    @classmethod
    def enable(cls, sample_period_s: float = 3.0) -> None:
        if cls._enabled:
            return
        cls._sample_period = max(0.5, float(sample_period_s)) if hasattr(cls, "_sample_period") else float(sample_period_s)
        try:
            cls._have_psutil = True
        except Exception:
            cls._have_psutil = False
        cls._enabled = True
        cls._stop = False
        if not cls._have_psutil:
            return
        def _sampler():
            p = psutil.Process(os.getpid())
            try:
                rss_mb = p.memory_info().rss / (1024**2)
                if rss_mb > getattr(cls, "_max_rss_mb", 0.0):
                    cls._max_rss_mb = float(rss_mb)
            except Exception:
                pass
            while not getattr(cls, "_stop", False):
                try:
                    rss_mb2 = p.memory_info().rss / (1024**2)
                    if rss_mb2 > getattr(cls, "_max_rss_mb", 0.0):
                        cls._max_rss_mb = float(rss_mb2)
                except Exception:
                    pass
                time.sleep(cls._sample_period)
        try:
            t = threading.Thread(target=_sampler, name="MemUtilSampler", daemon=True)
            t.start()
            cls._bg_thread = t
        except Exception:
            cls._bg_thread = None

    @classmethod
    def log_summary(cls, logger) -> None:
        rss_txt = "n/d"
        try:
            if getattr(cls, "_max_rss_mb", 0.0) > 0.0:
                rss_txt = f"{cls._max_rss_mb:.1f} MB"
        except Exception:
            pass
        logger.log(f"Pico RSS de memória: {rss_txt}")

    @classmethod
    def disable(cls) -> None:
        cls._stop = True
        cls._enabled = False

# ==================================================================================================
# --- 0. LOGGER ---
# ==================================================================================================
class Logger:
    def __init__(self, log_directory: str, log_filename: str = "resumo_execucao.log"):
        """
        Configura o sistema de logging ao ser instanciada.
        """
        # 1. Define o caminho completo e garante que o diretório exista
        self.log_file_path = os.path.join(log_directory, log_filename)
        os.makedirs(log_directory, exist_ok=True)

        # 2. Pega o logger e define o nível
        self.logger = logging.getLogger("automacao_logger")
        self.logger.setLevel(logging.INFO)

        # Evita adicionar handlers duplicados se a classe for instanciada mais de uma vez
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # 3. Configura o handler para escrever em arquivo com rotação
        handler = logging.handlers.RotatingFileHandler(
            self.log_file_path,
            maxBytes=2_000_000,
            backupCount=3,
            encoding='utf-8'
        )

        # 4. Adiciona um formato às mensagens
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)

        # 5. Adiciona o handler ao logger
        self.logger.addHandler(handler)

    def log(self, message: Any = ""):
        """
        Registra uma mensagem no console e no arquivo de log.
        """
        message_str = str(message)
        # Exibe no console para acompanhamento em tempo real
        print(message_str)
        sys.stdout.flush()
        # Envia a mensagem para o arquivo de log
        self.logger.info(message_str)

class LiveCounter:
    """
    Contador incremental com logs parciais por camada.
    Imprime quando se passa um intervalo mínimo de tempo ou quando forçado.
    """
    def __init__(self, logger, layer_num: int, min_seconds: float = 2.0):
        self.log = logger.log
        self.layer = int(layer_num)
        self.min_seconds = max(0.5, float(min_seconds))
        self.last_ts = time.perf_counter()
        self._tot = 0
        self._ciclo = 0
        self._dur = 0
        self._gapf = 0
        self._vales = 0
        self._hf = 0
        self._df = 0  # NOVO: Dominância de Fronteira
        self._aprov = 0

    def _should_print(self) -> bool:
        return (time.perf_counter() - self.last_ts) >= self.min_seconds

    def bump(self, stats_delta: Dict[str, int], aprov_delta: int = 0, force: bool = False) -> None:
        self._tot += int(stats_delta.get("cronogramas_testados", 0))
        self._ciclo += int(stats_delta.get("descartados_por_ciclo", 0))
        self._dur += int(stats_delta.get("descartados_por_duracao", 0))
        self._gapf += int(stats_delta.get("descartados_por_gap_futuro", 0))
        self._vales += int(stats_delta.get("descartados_por_vales", 0))
        self._hf += int(stats_delta.get("descartados_por_equivalencia_hist_fronteira", 0))
        self._df += int(stats_delta.get("descartados_por_dominancia_de_fronteira", 0))  # NOVO
        self._aprov += int(aprov_delta)
        if force or (time.perf_counter() - self.last_ts) >= self.min_seconds:
            self.log(
                f"    - Camada-{self.layer}"
                f" Testados: {self._tot:,} C: {self._ciclo:,} D: {self._dur:,}"
                f" GF: {self._gapf:,} V: {self._vales:,}"
                f" HF: {self._hf:,} DF: {self._df:,} Aprovado parcial: {self._aprov:,}"
            )
            self.last_ts = time.perf_counter()

# ==================================================================================================
# --- 1. CONFIGURAÇÃO ---
# ==================================================================================================
BASE_DIR = r"C:\Users\farmc\Downloads\res_automacao"

class Config:
    INPUT_FILE = os.path.join(BASE_DIR, "R38_tarefas_4.txt")
    _input_basename = os.path.splitext(os.path.basename(INPUT_FILE))[0]
    OUTPUT_FILE = os.path.join(BASE_DIR, f"{_input_basename}_resultados_unicos.txt")
    MULTIPROCESSING = True
    MAX_ARBITRARY_PREDECESSORS_PER_TASK = 1
    MAX_ALLOWED_UP_VALLEYS = 0
    ENABLE_FUTURE_GAP_PRUNING = 2 # 0 não verifica o GAP futuro outro número interio é a distância de verificação
    ENABLE_HF_PRUNING = True # Executa a poda por Histograma + Fronteira (HF) em cada camada    
    DURATION_TIER_RULE = False # Pode por nível de rede a parti da camada 1, altamente agressivo

    def compute_multiplier(self, base_duration_days):
        """
        Multiplicador dinâmico m(B), com B em meses:
        m(B) = m_min + (m_max - m_min) * min(1, B^(-alpha)).
        Parâmetros configuráveis via atributos opcionais: DM_MIN, DM_MAX, DM_ALPHA.
        Defaults: DM_MIN=1.20, DM_MAX=2.0, DM_ALPHA=0.506.
        """
        m_min = float(getattr(self, "DM_MIN", 1.20))
        m_max = float(getattr(self, "DM_MAX", 2.0))
        alpha = float(getattr(self, "DM_ALPHA", 0.506))
        b_months = max(1.0, float(base_duration_days)) / 30.44
        raw = m_min + (m_max - m_min) * min(1.0, b_months ** (-alpha))
        if raw < m_min:
            return m_min
        if raw > m_max:
            return m_max
        return raw

    def get_max_duration_multiplier(self, base_duration_days):
        """
        Retorna o multiplicador de prazo calculado pela política dinâmica interna.
        Elimina o uso de MAX_DURATION_MULTIPLIER fixo ou externo.
        """
        return float(self.compute_multiplier(base_duration_days))

    def get_multiprocessing_params(self) -> Tuple[int, int, int]:
        """
        Retorna (num_processos, chunksize, min_parents) com defaults sensatos.
        Pode ser sobrescrito via atributos: MP_PROCESSES, MP_CHUNKSIZE, MP_MIN_PARENTS;
        ou env: MP_PROCESSES, MP_CHUNKSIZE, MP_MIN_PARENTS.
        """
        def _env_int(name, default):
            try:
                return int(os.getenv(name, "").strip() or default)
            except Exception:
                return default
        procs = getattr(self, "MP_PROCESSES", _env_int("MP_PROCESSES", os.cpu_count() or 1))
        chunks = getattr(self, "MP_CHUNKSIZE", _env_int("MP_CHUNKSIZE", 1))
        min_parents = getattr(self, "MP_MIN_PARENTS", _env_int("MP_MIN_PARENTS", 64))
        return max(1, procs), max(1, chunks), max(1, min_parents)

_log_input_basename = os.path.splitext(os.path.basename(Config.INPUT_FILE))[0]
log_filename = f"{_log_input_basename}_resumo_execucao.log"
main_logger = Logger(log_directory=BASE_DIR, log_filename=log_filename)

# ==================================================================================================
# --- 2. ALGORITMOS AUXILIARES DE GRAFO E MÉTRICAS ---
# ==================================================================================================

def make_hist_signature(
    finish_times: Dict[str, int],
    durations: Dict[str, int],
    resources: Dict[str, Tuple[str, int]],
    tasks_in_scope: List[str]
) -> Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...]:
    """
    Assinatura de histograma por varredura de eventos (sem vetor diário).
    Preserva exatamente o mesmo RLE que seria obtido a partir da linha do tempo densa.
    """
    if not tasks_in_scope:
        return tuple()

    project_duration = 0
    for tid in tasks_in_scope:
        if tid in finish_times:
            project_duration = max(project_duration, finish_times[tid])
    if project_duration <= 0:
        return tuple()

    events_by_res: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for tid in tasks_in_scope:
        if tid in finish_times and tid in durations and tid in resources:
            res_id, _ = resources[tid]
            if not res_id:
                continue
            ft = finish_times[tid]
            st = max(0, ft - durations[tid])
            events_by_res[res_id][st] += 1
            events_by_res[res_id][ft] -= 1

    signature_items = []
    for res_id in sorted(events_by_res.keys()):
        events = events_by_res[res_id]
        # sentinela para fechar o último segmento
        if project_duration not in events:
            events[project_duration] += 0
        cur = 0
        last_t = 0
        runs: List[Tuple[int, int]] = []
        for t in sorted(events.keys()):
            seg_len = t - last_t
            if seg_len > 0:
                # adiciona o trecho constante [last_t, t) com valor 'cur'
                if runs and runs[-1][0] == cur:
                    runs[-1] = (cur, runs[-1][1] + seg_len)
                else:
                    runs.append((cur, seg_len))
            cur += events[t]
            last_t = t
        signature_items.append((res_id, tuple(runs)))

    return tuple(signature_items)

def make_frontier_signature(
    dependencies: Dict[str, List[str]],
    finish_times: Dict[str, int],
    tasks_done_scope: Set[str],
    tasks_future: List[str]
) -> Tuple[Tuple[str, int, int], ...]:
    """
    Assinatura de 'fronteira' para garantir completude ao deduplicar por camada.
    Para cada tarefa futura (camadas > L), captura:
      - quantos predecessores dela já estão concluídos dentro do escopo (<= L)
      - o maior finish_time desses predecessores já concluídos

    Isso preserva a "visão" que as tarefas futuras têm do estado atual.
    Retorna tupla ordenada por task_id:
      ((task_id, n_preds_concluidos_no_escopo, max_ft_preds_no_escopo), ...)
    """
    items = []
    for t in sorted(tasks_future):
        preds = dependencies.get(t, [])
        done_preds = [p for p in preds if (p in tasks_done_scope and p in finish_times)]
        n_done = len(done_preds)
        max_ft = max((finish_times[p] for p in done_preds), default=0)
        items.append((t, n_done, max_ft))
    return tuple(items)

def calculate_earliest_start_times(pred_graph: Dict[str, List[str]], durations: Dict[str, int], task_ids: List[str]) -> Optional[Dict[str, int]]:
    start_times, successors_graph, in_degree = {tid: 0 for tid in task_ids}, defaultdict(list), {tid: 0 for tid in task_ids}
    for task_id, preds in pred_graph.items():
        if task_id not in task_ids: continue
        for p_id in preds:
            if p_id in task_ids:
                successors_graph[p_id].append(task_id); in_degree[task_id] += 1
    queue, processed_count = deque([tid for tid in task_ids if in_degree[tid] == 0]), 0
    while queue:
        current_task_id = queue.popleft(); processed_count += 1
        finish_time = start_times[current_task_id] + durations[current_task_id]
        for successor_id in successors_graph.get(current_task_id, []):
            start_times[successor_id] = max(start_times.get(successor_id, 0), finish_time)
            in_degree[successor_id] -= 1
            if in_degree[successor_id] == 0: queue.append(successor_id)
    return start_times if processed_count == len(task_ids) else None

def calculate_resource_valleys(
    start_times: Dict[str, int],
    durations: Dict[str, int],
    resources: Dict[str, Tuple[str, int]],
    tasks_in_scope: List[str],
    limit: Optional[int] = None
) -> int:
    if not tasks_in_scope:
        return 0
    events_by_res: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for tid in tasks_in_scope:
        if tid in start_times and tid in durations:
            res_id, _ = resources.get(tid, (None, 0))
            if not res_id:
                continue
            st = start_times[tid]
            ft = st + durations[tid]
            if ft > st:
                events_by_res[res_id][st] += 1
                events_by_res[res_id][ft] -= 1
    if isinstance(limit, int) and limit <= 0:
        for events in events_by_res.values():
            prev_dir = 0
            for t in sorted(events.keys()):
                delta = events[t]
                if delta == 0:
                    continue
                curr_dir = 1 if delta > 0 else -1
                if prev_dir == -1 and curr_dir == 1:
                    return 1
                prev_dir = curr_dir
        return 0
    total = 0
    for events in events_by_res.values():
        prev_dir = 0
        for t in sorted(events.keys()):
            delta = events[t]
            if delta == 0:
                continue
            curr_dir = 1 if delta > 0 else -1
            if prev_dir == -1 and curr_dir == 1:
                total += 1
                if isinstance(limit, int) and limit > 0 and total > limit:
                    return total
            prev_dir = curr_dir
    return total

# ==================================================================================================
# --- 3. ESTRUTURAS DE DADOS PRINCIPAIS ---
# ==================================================================================================
class ProjectData:
    def __init__(self, config: Config):
        self.config = config
        self.all_tasks_info: List[Dict[str, Any]] = []
        self.task_ids: List[str] = []
        self.durations: Dict[str, int] = {}
        self.resources: Dict[str, Tuple[str, int]] = {}
        self.mandatory_graph: Dict[str, List[str]] = {}
        self._load_tasks_from_file()

        self.base_duration, self.max_duration_allowed = 0, 0
        self._calculate_base_duration()
        self.descendants_map: Dict[str, Set[str]] = self._calculate_descendants_map()
        self.task_layers: Dict[int, List[str]] = self._calculate_task_layers()
        self.resource_to_tasks_map: Dict[str, List[str]] = self._map_resources_to_tasks()
        self.task_to_layer_map: Dict[str, int] = {
            task: layer_num for layer_num, tasks in self.task_layers.items() for task in tasks
        }
        self.networks = self._identify_networks()
        self.network_durations = self._calculate_network_durations()
        self.task_to_network_duration = self._build_task_to_duration_map()

        # >>> NOVO: superblocos por recurso (um bloco por recurso)
        self.blocks_by_res, self.block_weight, self.task_to_block = self._build_resource_superblocks()
        self.block_rank, self.task_block_rank = self._rank_resource_blocks(self.blocks_by_res, self.block_weight)
        self._log_blocks_ranking()  # agora no formato “Bloco i (res, X dias-recurso): N tarefas”

        # Logs já existentes
        self._log_pre_analysis_summaries()

        # Poda por GAP futuro (já existente)
        self.prunable_resources_info: Dict[str, Dict] = self._find_prunable_resources_by_gap()

    def _load_tasks_from_file(self):
        with open(self.config.INPUT_FILE, 'r', encoding='utf-8') as f: content = f.read().strip()
        file_content_str = content.split(']', 1)[0] + ']'
        list_start_index = file_content_str.find('[(')
        if list_start_index == -1: raise ValueError("Could not find the start of the task list.")
        list_str = file_content_str[list_start_index:]
        try: raw_tasks = ast.literal_eval(list_str)
        except (ValueError, SyntaxError) as e: raise ValueError(f"Failed to parse tasks from input file: {e}")
        for t in raw_tasks:
            task_id, duration = t[0], t[1]
            preds = [p.strip() for p in t[2].split(',') if p.strip()] if len(t) > 2 and t[2] else []
            self.all_tasks_info.append({"id": task_id, "duration": duration, "mandatory_preds": preds})
            self.task_ids.append(task_id)
            self.durations[task_id], self.mandatory_graph[task_id] = duration, preds
            if len(t) > 4 and t[4]: self.resources[task_id] = (t[4], t[5] if len(t) > 5 else 0)
        self.task_ids.sort()

    def _identify_networks(self) -> Dict[int, List[str]]:
        networks, visited, graph = {}, set(), defaultdict(set)
        for task, preds in self.mandatory_graph.items():
            for pred in preds:
                graph[task].add(pred)
                graph[pred].add(task)
        nid_counter = 0
        for task_id in self.task_ids:
            if task_id not in visited:
                nid_counter += 1
                component = []
                q = deque([task_id]); visited.add(task_id)
                while q:
                    curr = q.popleft(); component.append(curr)
                    for neighbor in graph[curr]:           # <- apenas esta linha
                        if neighbor not in visited:
                            visited.add(neighbor); q.append(neighbor)
                networks[nid_counter] = sorted(component)
        return networks

    def _calculate_base_duration(self):
        main_logger.log("\n--- ETAPA PRELIMINAR: Análise do Cronograma Base 0 ---")
        base_starts = calculate_earliest_start_times(self.mandatory_graph, self.durations, self.task_ids)
        if not base_starts:
            raise ValueError("O cronograma base é inválido ou cíclico.")
        self.base_duration = max((base_starts[tid] + self.durations[tid] for tid in self.task_ids), default=0)
        m = self.config.get_max_duration_multiplier(self.base_duration)
        self.max_duration_allowed = int(math.ceil(self.base_duration * m))
        main_logger.log(f"  1. Duração do cronograma base: {self.base_duration} dias.")
        main_logger.log(f"  2. Duração máxima permitida (calculada): {self.max_duration_allowed} dias.")

    def _log_pre_analysis_summaries(self):
        layer_summary = "".join([f"\n      - Camada {num} ({len(tasks)} tarefas)" for num, tasks in sorted(self.task_layers.items())])
        main_logger.log(f"  4. Análise Estrutural: {layer_summary}")
        main_logger.log("  5. Ranking de Redes por Duração (Caminho Crítico Interno):")
        duration_groups = defaultdict(list)
        for nid, dur in self.network_durations.items():
            duration_groups[dur].append(nid)
        sorted_tiers = sorted(duration_groups.items(), key=lambda item: item[0], reverse=True)
        tier_num = 1
        for duration, nids in sorted_tiers:
            num_tasks_in_tier = sum(len(self.networks[nid]) for nid in nids)
            main_logger.log(f"     - Tier {tier_num} ({duration} dias, {num_tasks_in_tier} tarefas): Redes {sorted(nids)}")
            tier_num += 1

    def _log_blocks_ranking(self) -> None:
        """
        Loga o ranking em formato compacto, agregando por recurso:
        - Bloco 1 (mec, 16 dias-recurso): 6 tarefas.
        - Bloco 2 (est, 4 dias-recurso): 4 tarefas.
        ...
        """
        try:
            items = []
            for res_id, blist in self.blocks_by_res.items():
                if not blist:
                    continue
                bloco = blist[0]
                bid = (res_id, 1)
                w = int(self.block_weight.get(bid, 0))
                items.append((res_id, w, len(bloco)))
            items.sort(key=lambda x: x[1], reverse=True)

            main_logger.log("  3. Ranking de Blocos de Recursos por dias-recurso:")
            for i, (res_id, w, n) in enumerate(items, start=1):
                main_logger.log(f"      - Bloco {i} ({res_id}, {w} dias-recurso): {n} tarefas.")
        except Exception:
            pass

    def _find_prunable_resources_by_gap(self) -> Dict[str, Dict]:
        """
        Seleciona recursos elegíveis para a verificação de 'gap futuro', usando
        apenas ENABLE_FUTURE_GAP_PRUNING como parâmetro:
        - 0  => poda desabilitada (retorna {}).
        - >0 => distância mínima de camadas entre usos do recurso.
        """
        prunable_info: Dict[str, Dict] = {}
        threshold = int(getattr(self.config, "ENABLE_FUTURE_GAP_PRUNING", 0) or 0)

        if threshold <= 0:
            main_logger.log("  6. Poda por Gap Futuro desabilitada (ENABLE_FUTURE_GAP_PRUNING=0).")
            return prunable_info

        main_logger.log(f"  6. Analisando Recursos para Poda por Gap Futuro (distância mínima={threshold}) ---")

        for res_id, tasks in self.resource_to_tasks_map.items():
            if len(tasks) < 2:
                continue

            layers_used = sorted({self.task_to_layer_map[tid] for tid in tasks})
            if len(layers_used) <= 1:
                continue

            layer_distance = layers_used[-1] - layers_used[0]
            if layer_distance < threshold:
                continue

            early_layer, late_layer = layers_used[0], layers_used[-1]
            early_tasks = {tid for tid in tasks if self.task_to_layer_map[tid] == early_layer}
            late_tasks = {tid for tid in tasks if self.task_to_layer_map[tid] == late_layer}

            # Exige dependência causal de alguma 'early' para alguma 'late'
            has_causal = any(
                (lt in self.descendants_map.get(et, set()))
                for et in early_tasks for lt in late_tasks
            )
            if not has_causal:
                continue

            main_logger.log(
                f"     - Recurso '{res_id}' identificado para verificação (camadas {early_layer}->{late_layer})."
            )
            prunable_info[res_id] = {"early_tasks": early_tasks, "late_tasks": late_tasks}

        if prunable_info:
            recursos = sorted(prunable_info.keys())
            main_logger.log(
                f"     - Verificação LOCAL ASCENDENTE habilitada para recursos: {recursos} (distância mínima={threshold})."
            )
        else:
            main_logger.log("    - Nenhum recurso encontrado para esta otimização de poda.")

        return prunable_info

    def _calculate_task_layers(self) -> Dict[int, List[str]]:
        layers, successors, in_degree = {tid: -1 for tid in self.task_ids}, defaultdict(list), {tid: len(self.mandatory_graph.get(tid, [])) for tid in self.task_ids}
        for tid, preds in self.mandatory_graph.items():
            for p_id in preds:
                if p_id in self.task_ids: successors[p_id].append(tid)
        queue = deque([tid for tid, deg in in_degree.items() if deg == 0])
        for tid in queue: layers[tid] = 0
        head = 0
        while head < len(queue):
            u = queue[head]; head += 1
            for v in successors[u]:
                layers[v] = max(layers.get(v, -1), layers[u] + 1)
                in_degree[v] -= 1
                if in_degree[v] == 0: queue.append(v)
        grouped_layers = defaultdict(list)
        for task, layer in sorted(layers.items()):
            if layer != -1: grouped_layers[layer].append(task)
        return dict(sorted(grouped_layers.items()))

    def _calculate_descendants_map(self) -> Dict[str, Set[str]]:
        successors = defaultdict(list)
        for tid, preds in self.mandatory_graph.items():
            for p_id in preds:
                if p_id in self.task_ids: successors[p_id].append(tid)
        descendants_map = {}
        for tid in self.task_ids:
            descendants, q, visited = set(), deque(successors.get(tid, [])), set()
            while q:
                curr = q.popleft()
                if curr in visited: continue
                visited.add(curr); descendants.add(curr)
                q.extend(successors.get(curr, []))
            descendants_map[tid] = descendants
        return descendants_map

    def _calculate_network_durations(self) -> Dict[int, int]:
        durations = {}
        for nid, tasks in self.networks.items():
            local_mandatory_graph = {t: [p for p in self.mandatory_graph.get(t, []) if p in tasks] for t in tasks}
            start_times = calculate_earliest_start_times(local_mandatory_graph, self.durations, tasks)
            if start_times:
                max_finish_time = max((st + self.durations[tid] for tid, st in start_times.items()), default=0)
                durations[nid] = max_finish_time
            else:
                durations[nid] = sum(self.durations[t] for t in tasks)
        return durations

    def _map_resources_to_tasks(self) -> Dict[str, List[str]]:
        res_map = defaultdict(list)
        for task_id, (res_id, _) in self.resources.items(): res_map[res_id].append(task_id)
        return res_map

    def _build_resource_superblocks(self) -> Tuple[Dict[str, List[List[str]]], Dict[Tuple[str, int], int], Dict[str, Tuple[str, int]]]:
        """
        Superblocos por recurso: UM bloco por recurso contendo TODAS as tarefas desse recurso.
        Retorna:
        - blocks_by_res: {res_id -> [[t1, t2, ...]]} (lista com 1 bloco por recurso)
        - block_weight: {(res_id, 1) -> soma das durações do recurso}
        - task_to_block: {task_id -> (res_id, 1)}
        """
        blocks_by_res: Dict[str, List[List[str]]] = {}
        block_weight: Dict[Tuple[str, int], int] = {}
        task_to_block: Dict[str, Tuple[str, int]] = {}

        for res_id, tasks in self.resource_to_tasks_map.items():
            ordered_tasks = list(tasks)  # já vem por ID; topologia não importa para o superbloco
            blocks_by_res[res_id] = [ordered_tasks]
            bid = (res_id, 1)
            w = sum(int(self.durations.get(t, 0)) for t in ordered_tasks)
            block_weight[bid] = int(w)
            for t in ordered_tasks:
                task_to_block[t] = bid

        return blocks_by_res, block_weight, task_to_block

    def _build_resource_blocks(self) -> Tuple[Dict[str, List[List[str]]], Dict[Tuple[str, int], int], Dict[str, Tuple[str, int]]]:
        """
        Cria blocos por recurso: cada bloco é uma cadeia máxima (in-degree<=1 e out-degree<=1
        no grafo induzido pelas predecessoras obrigatórias do mesmo recurso).
        Retorna:
        - blocks_by_res: {res_id -> [ [task_ids do bloco 1], [task_ids do bloco 2], ... ]}
        - block_weight: {(res_id, idx) -> soma das durações do bloco}
        - task_to_block: {task_id -> (res_id, idx)}
        """
        blocks_by_res: Dict[str, List[List[str]]] = defaultdict(list)
        block_weight: Dict[Tuple[str, int], int] = {}
        task_to_block: Dict[str, Tuple[str, int]] = {}

        for res_id, tasks in self.resource_to_tasks_map.items():
            if not tasks:
                continue
            tasks_set = set(tasks)
            preds_in_res = {t: [p for p in self.mandatory_graph.get(t, []) if p in tasks_set] for t in tasks}
            succs_in_res: Dict[str, List[str]] = defaultdict(list)
            indeg: Dict[str, int] = {}
            for t in tasks:
                indeg[t] = len(preds_in_res[t])
                for p in preds_in_res[t]:
                    succs_in_res[p].append(t)

            visited: Set[str] = set()
            # inicia blocos em "pontas" e nós de ramificação
            for t in tasks:
                if t in visited:
                    continue
                if indeg[t] != 1 or len(succs_in_res.get(t, [])) != 1:
                    cur, bloco = t, []
                    while cur not in visited:
                        visited.add(cur); bloco.append(cur)
                        nxts = succs_in_res.get(cur, [])
                        if len(nxts) == 1 and indeg.get(nxts[0], 0) == 1 and len(succs_in_res.get(cur, [])) == 1:
                            cur = nxts[0]
                        else:
                            break
                    if bloco:
                        blocks_by_res[res_id].append(bloco)

            # tarefas isoladas que sobraram viram blocos unitários
            for t in tasks:
                if t not in visited:
                    blocks_by_res[res_id].append([t])

            # pesos e mapas
            for idx, bloco in enumerate(blocks_by_res[res_id], start=1):
                bid = (res_id, idx)
                w = sum(self.durations.get(u, 0) for u in bloco)
                block_weight[bid] = int(w)
                for u in bloco:
                    task_to_block[u] = bid

        return dict(blocks_by_res), block_weight, task_to_block

    def _build_task_to_duration_map(self) -> Dict[str, int]:
        task_map = {}
        for nid, tasks in self.networks.items():
            duration = self.network_durations.get(nid, 0)
            for task in tasks:
                task_map[task] = duration
        return task_map

    def _rank_resource_blocks(self, blocks_by_res: Dict[str, List[List[str]]],
                            block_weight: Dict[Tuple[str, int], int]) -> Tuple[Dict[Tuple[str, int], int], Dict[str, int]]:
        """
        Ordena blocos por peso (dias-recurso) desc. e devolve:
        - block_rank: {(res_id, idx) -> posição 1..K}
        - task_block_rank: {task_id -> rank do bloco da tarefa}
        """
        all_blocks = []
        for res_id, blist in blocks_by_res.items():
            for idx, _ in enumerate(blist, start=1):
                bid = (res_id, idx)
                all_blocks.append((bid, block_weight.get(bid, 0)))
        all_blocks.sort(key=lambda x: x[1], reverse=True)

        block_rank: Dict[Tuple[str, int], int] = {}
        for pos, (bid, _) in enumerate(all_blocks, start=1):
            block_rank[bid] = pos

        task_block_rank: Dict[str, int] = {}
        for res_id, blist in blocks_by_res.items():
            for idx, bloco in enumerate(blist, start=1):
                rank = block_rank.get((res_id, idx), 10**9)
                for t in bloco:
                    task_block_rank[t] = rank
        return block_rank, task_block_rank

# ==================================================================================================
# --- 4. ORQUESTRADOR PRINCIPAL ---
# ==================================================================================================
class Scheduler:
    def __init__(self, project_data: ProjectData, config: Config):
        self.project_data = project_data
        self.config = config
        self.solutions: List[Dict[str, Any]] = []
        self.total_stats = defaultdict(int)
        self.layer_0_solutions: List[Dict[str, Any]] = []

    def _task_ids_ordered(self) -> List[str]:
        if not hasattr(self, "_cached_task_ids_ordered"):
            setattr(self, "_cached_task_ids_ordered", list(self.project_data.task_ids))
        return getattr(self, "_cached_task_ids_ordered")

    def _pack_start_times_vec(self, start_times: Dict[str, int]) -> Tuple[int, ...]:
        tids = self._task_ids_ordered()
        return tuple(start_times.get(t, 0) for t in tids)

    def _start_times_from_vec(self, vec: Tuple[int, ...]) -> Dict[str, int]:
        tids = self._task_ids_ordered()
        return {tids[i]: vec[i] for i in range(len(tids)) if vec[i] != 0}

    def _ensure_compact_solution(self, sol: Dict[str, Any]) -> Dict[str, Any]:
        if "start_times_vec" not in sol:
            st = sol.get("start_times", {})
            sol["start_times_vec"] = self._pack_start_times_vec(st)
            if "start_times" in sol:
                del sol["start_times"]
        return sol

    def _get_start_times(self, sol: Dict[str, Any]) -> Dict[str, int]:
        if "start_times_vec" in sol:
            return self._start_times_from_vec(sol["start_times_vec"])
        return sol.get("start_times", {})

    def run_optimization(self) -> None:
        initial_solution = {"dependencies": self.project_data.mandatory_graph.copy(), "id": "0"}
        cumulative_solutions = [initial_solution]
        tasks_from_previous_layers: List[str] = []
        self.total_stats = defaultdict(int)
        self._survivors_per_layer: List[int] = []

        for layer_num in sorted(self.project_data.task_layers.keys()):
            layer_tasks = sorted(self.project_data.task_layers[layer_num])
            main_logger.log(f"\n--- ETAPA {1 + layer_num}: Processando Camada {layer_num} ---")

            tasks_in_scope_for_preds = tasks_from_previous_layers + layer_tasks
            layer_task_options = self._generate_predecessor_options(layer_tasks, tasks_in_scope_for_preds)

            cumulative_solutions, layer_stats = self._process_layer(
                parent_solutions=cumulative_solutions,
                layer_task_options=layer_task_options
            )

            for k, v in layer_stats.items():
                self.total_stats[k] += v

            # Registrar aprovados desta camada (após dedup intra-camada)
            self._survivors_per_layer.append(len(cumulative_solutions))

            if layer_num == 0:
                for i, sol in enumerate(cumulative_solutions, 1):
                    sol['id'] = str(i)
                self.layer_0_solutions = [sol.copy() for sol in cumulative_solutions]

            main_logger.log(f"  Encontrados {len(cumulative_solutions):,} estados ótimos cumulativos para a próxima etapa.")
            self._save_intermediate_results(cumulative_solutions, layer_num)
            tasks_from_previous_layers.extend(layer_tasks)

            if not cumulative_solutions:
                main_logger.log("\n  Nenhuma solução viável encontrada nesta etapa. Interrompendo.")
                break

        self.solutions = cumulative_solutions

    def _find_non_dominated_subset(
            self,
            solution_group: List[Dict[str, Any]],
            tasks_done_set: Set[str],
            downstream_tasks: List[str]
        ) -> List[Dict[str, Any]]:
        if len(solution_group) <= 1:
            return solution_group
        preds_by_task = {
            t: [p for p in self.project_data.mandatory_graph.get(t, []) if p in tasks_done_set]
            for t in downstream_tasks
        }
        c_by_task = {t: len(preds_by_task[t]) for t in downstream_tasks}
        frontier: List[Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]] = []
        for sol in solution_group:
            sol = self._ensure_compact_solution(sol)
            ft = self._compute_finish_times(self._get_start_times(sol))
            cand_vec = {t: (c_by_task[t], max((ft.get(p, 0) for p in preds_by_task[t]), default=0)) for t in downstream_tasks}
            dominated = False
            remove_idx: List[int] = []
            for idx, (_, vec) in enumerate(frontier):
                dominates = True
                better = False
                for t in downstream_tasks:
                    c_o, t_o = vec[t]; c_c, t_c = cand_vec[t]
                    if c_o < c_c or t_o > t_c:
                        dominates = False
                        break
                    if c_o > c_c or t_o < t_c:
                        better = True
                if dominates and better:
                    dominated = True
                    break
                dominates_other = True
                better2 = False
                for t in downstream_tasks:
                    c_o, t_o = vec[t]; c_c, t_c = cand_vec[t]
                    if c_c < c_o or t_c > t_o:
                        dominates_other = False
                        break
                    if c_c > c_o or t_c < t_o:
                        better2 = True
                if dominates_other and better2:
                    remove_idx.append(idx)
            if not dominated:
                if remove_idx:
                    for i in reversed(sorted(remove_idx)):
                        frontier.pop(i)
                frontier.append((sol, cand_vec))
        return [s for s, _ in frontier]

    def _prune_by_dominance_groups(
            self,
            solutions: List[Dict[str, Any]],
            tasks_scope_done: List[str],
            tasks_done_set: Set[str],
            downstream_tasks: List[str]
        ) -> List[Dict[str, Any]]:
        if not solutions:
            return []
        survivors_by_hist: Dict[str, List[Dict[str, Any]]] = {}
        for sol in solutions:
            sol = self._ensure_compact_solution(sol)
            ft = self._compute_finish_times(self._get_start_times(sol))
            h_sig = make_hist_signature(
                ft,
                self.project_data.durations,
                self.project_data.resources,
                tasks_scope_done
            )
            h_key = str(h_sig)
            group = survivors_by_hist.get(h_key)
            if group is None:
                survivors_by_hist[h_key] = [sol]
                continue
            updated = self._find_non_dominated_subset(group + [sol], tasks_done_set, downstream_tasks)
            survivors_by_hist[h_key] = updated
        out: List[Dict[str, Any]] = []
        for lst in survivors_by_hist.values():
            out.extend(lst)
        return out

    def _resolve_duration_tier_rule(self) -> Optional[Callable[[int, int], bool]]:
        """
        Resolve DURATION_TIER_RULE para:
        - None: sem poda
        - True: regra padrão (send >= recv)
        - callable: regra personalizada (normalizada para bool)
        """
        raw = getattr(self.config, "DURATION_TIER_RULE", None)
        if raw in (None, False):
            return None
        if raw is True:
            return lambda recv, send: send >= recv
        if callable(raw):
            def _wrapped(recv: int, send: int) -> bool:
                return bool(raw(recv, send))
            return _wrapped
        return None
    
    def _filter_candidates_by_tier(
        self,
        task_id: str,
        base: List[str],
        rule: Optional[Callable[[int, int], bool]]
    ) -> List[str]:
        """
        Aplica (ou não) a poda por nível de rede para a task_id em base.
        """
        if rule is None:
            return base
        durations = self.project_data.task_to_network_duration
        recv = durations.get(task_id, 0)
        out: List[str] = []
        for p in base:
            if rule(recv, durations.get(p, 0)):
                out.append(p)
        return out

    def _build_predecessor_combos(self, candidates: List[str]) -> List[Tuple[str, ...]]:
        """
        Gera combinações ordenadas até MAX_ARBITRARY_PREDECESSORS_PER_TASK.
        """
        opts: List[Tuple[str, ...]] = []
        maxk = self.config.MAX_ARBITRARY_PREDECESSORS_PER_TASK
        for r in range(maxk + 1):
            for combo in itertools.combinations(candidates, r):
                opts.append(tuple(sorted(combo)))
        return sorted(opts)

    def _generate_predecessor_options(
        self,
        layer_tasks: List[str],
        tasks_in_scope: List[str]
    ) -> Dict[str, List[Tuple[str, ...]]]:
        options: Dict[str, List[Tuple[str, ...]]] = {}
        durations = self.project_data.task_to_network_duration
        layer_map = self.project_data.task_to_layer_map
        layer_num = layer_map.get(layer_tasks[0], -1)

        # Camada 0 — mantém lógica original (dinâmica e hierárquica)
        if layer_num == 0:
            main_logger.log("  Aplicando lógica de CAMADA e NÍVEL DE REDE para a Camada 0.")
            head = {t for t in layer_tasks if t in durations}
            for task_id in head:
                recv = durations[task_id]
                cands: List[str] = []
                for p_id in head:
                    if p_id == task_id:
                        continue
                    if task_id in self.project_data.descendants_map.get(p_id, set()):
                        continue
                    if durations[p_id] >= recv:
                        cands.append(p_id)
                options[task_id] = self._build_predecessor_combos(cands)
            return options

        # Camadas >= 1 — regra base + poda por nível (se aplicável)
        typed_rule = self._resolve_duration_tier_rule()
        log_suffix = "ATIVA" if typed_rule else "DESLIGADA"
        if layer_num == 1:
            main_logger.log("  Aplicando lógica de CAMADA e NÍVEL DE REDE para a Camada 1 (poda por nível: %s)." % log_suffix)
        else:
            main_logger.log("  Aplicando lógica de CAMADA e NÍVEL DE REDE subsequentes (poda por nível: %s)." % log_suffix)

        for task_id in layer_tasks:
            descendants = self.project_data.descendants_map.get(task_id, set())
            base = [p for p in tasks_in_scope if p != task_id and p not in descendants]
            cands = self._filter_candidates_by_tier(task_id, base, typed_rule)
            options[task_id] = self._build_predecessor_combos(cands)

        return options

    def _process_layer(
            self,
            parent_solutions: List[Dict[str, Any]],
            layer_task_options: Dict[str, List[Tuple[str, ...]]]
        ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        if not parent_solutions:
            return [], defaultdict(int)

        layer_num, tasks_scope_done, tasks_scope_future, tasks_done_set = self._build_layer_scopes(layer_task_options)
        main_logger.log(f"  Avaliando a partir de {len(parent_solutions):,} estados da camada anterior.")
        main_logger.log("  Poda por Histograma + Fronteira: " + ("ATIVADA" if self.config.ENABLE_HF_PRUNING else "DESLIGADA"))

        use_mp = self.config.MULTIPROCESSING
        procs, chunks, min_parents = self.config.get_multiprocessing_params()
        main_logger.log(f"  Multiprocessamento: {'ATIVADO' if use_mp else 'DESLIGADO'} (proc={procs}, chunk={chunks}, limiar={min_parents})")

        layer_stats: Dict[str, int] = defaultdict(int)
        unique_keys: Set[str] = set()
        basekey_by_id: Dict[int, str] = {}
        frontier_by_hist: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]]] = {}
        frontier_ctx = self._frontier_ctx_make(tasks_done_set, tasks_scope_future)
        live = LiveCounter(main_logger, layer_num, min_seconds=2.0)

        if use_mp and len(parent_solutions) >= min_parents and procs > 1:
            iterator = self._iter_parent_results_mp(parent_solutions, layer_task_options, tasks_scope_done, procs, chunks, min_parents)
        else:
            iterator = self._iter_parent_results_seq(parent_solutions, layer_task_options, tasks_scope_done)

        for sols, stats in iterator:
            for k, v in stats.items():
                layer_stats[k] += v

            aprov_delta_batch = 0
            hf_delta_batch = 0
            df_delta_batch = 0

            for sol in sols:
                aprov_d, hf_d, df_d = self._online_frontier_update(
                    sol,
                    frontier_by_hist,
                    unique_keys,
                    basekey_by_id,
                    frontier_ctx,
                    tasks_scope_done,
                    tasks_done_set,
                    tasks_scope_future
                )
                aprov_delta_batch += aprov_d
                hf_delta_batch += hf_d
                df_delta_batch += df_d

            if hf_delta_batch:
                layer_stats["descartados_por_equivalencia_hist_fronteira"] += hf_delta_batch
            if df_delta_batch:
                layer_stats["descartados_por_dominancia_de_fronteira"] += df_delta_batch

            live.bump({
                **stats,
                "descartados_por_equivalencia_hist_fronteira": hf_delta_batch,
                "descartados_por_dominancia_de_fronteira": df_delta_batch
            }, aprov_delta_batch)

        survivors: List[Dict[str, Any]] = []
        for lst in frontier_by_hist.values():
            survivors.extend([s for s, _ in lst])

        self._log_layer_statistics(layer_num, layer_stats)
        return survivors, layer_stats

    def _build_layer_scopes(self, layer_task_options):
        """
        Retorna (current_layer_num, tasks_scope_done, tasks_scope_future, tasks_done_set)
        com base nas camadas (<=L e >L).
        """
        current_layer_tasks = sorted(layer_task_options.keys())
        if not current_layer_tasks:
            return 0, [], [], set()

        any_task = current_layer_tasks[0]
        current_layer_num = self.project_data.task_to_layer_map.get(any_task, 0)

        tasks_scope_done, tasks_scope_future = [], []
        for lyr, tasks in self.project_data.task_layers.items():
            if lyr <= current_layer_num:
                tasks_scope_done.extend(tasks)
            else:
                tasks_scope_future.extend(tasks)
        return current_layer_num, tasks_scope_done, tasks_scope_future, set(tasks_scope_done)

    def _dedupe_by_hist_and_frontier(self, sols, tasks_scope_done, tasks_done_set, tasks_scope_future, out_dict):
        for sol in sols:
            start_times = sol['start_times']
            finish_times = self._compute_finish_times(start_times)
            h_sig = make_hist_signature(
                finish_times=finish_times,
                durations=self.project_data.durations,
                resources=self.project_data.resources,
                tasks_in_scope=tasks_scope_done
            )
            f_sig = make_frontier_signature(
                dependencies=sol['dependencies'],
                finish_times=finish_times,
                tasks_done_scope=tasks_done_set,
                tasks_future=tasks_scope_future
            )
            base_key = self._digest_pair_signature(h_sig, f_sig)
            existing = out_dict.get(base_key)
            if existing is None:
                out_dict[base_key] = sol
                continue
            ex_ft = self._compute_finish_times(existing['start_times'])
            ex_h = make_hist_signature(ex_ft, self.project_data.durations, self.project_data.resources, tasks_scope_done)
            ex_f = make_frontier_signature(existing['dependencies'], ex_ft, tasks_done_set, tasks_scope_future)
            if (ex_h, ex_f) == (h_sig, f_sig):
                continue
            suffix = 1
            while True:
                k2 = f"{base_key}#{suffix}"
                other = out_dict.get(k2)
                if other is None:
                    out_dict[k2] = sol
                    break
                o_ft = self._compute_finish_times(other['start_times'])
                o_h = make_hist_signature(o_ft, self.project_data.durations, self.project_data.resources, tasks_scope_done)
                o_f = make_frontier_signature(other['dependencies'], o_ft, tasks_done_set, tasks_scope_future)
                if (o_h, o_f) == (h_sig, f_sig):
                    break
                suffix += 1

    def _ingest_solution_with_dedup(
        self,
        sol: Dict[str, Any],
        tasks_scope_done: List[str],
        tasks_done_set: Set[str],
        tasks_scope_future: List[str],
        out_dict: Dict[str, Dict[str, Any]]
    ) -> bool:
        """
        Insere 'sol' em out_dict aplicando deduplicação por (Histograma, Fronteira).
        Retorna True se 'sol' foi mantida; False se era equivalente a alguma existente.
        """
        start_times = sol['start_times']
        finish_times = self._compute_finish_times(start_times)
        h_sig = make_hist_signature(
            finish_times=finish_times,
            durations=self.project_data.durations,
            resources=self.project_data.resources,
            tasks_in_scope=tasks_scope_done
        )
        f_sig = make_frontier_signature(
            dependencies=sol['dependencies'],
            finish_times=finish_times,
            tasks_done_scope=tasks_done_set,
            tasks_future=tasks_scope_future
        )
        base_key = self._digest_pair_signature(h_sig, f_sig)
        existing = out_dict.get(base_key)
        if existing is None:
            out_dict[base_key] = sol
            return True
        ex_ft = self._compute_finish_times(existing['start_times'])
        ex_h = make_hist_signature(ex_ft, self.project_data.durations, self.project_data.resources, tasks_scope_done)
        ex_f = make_frontier_signature(existing['dependencies'], ex_ft, tasks_done_set, tasks_scope_future)
        if (ex_h, ex_f) == (h_sig, f_sig):
            return False
        suffix = 1
        while True:
            k2 = f"{base_key}#{suffix}"
            other = out_dict.get(k2)
            if other is None:
                out_dict[k2] = sol
                return True
            o_ft = self._compute_finish_times(other['start_times'])
            o_h = make_hist_signature(o_ft, self.project_data.durations, self.project_data.resources, tasks_scope_done)
            o_f = make_frontier_signature(other['dependencies'], o_ft, tasks_done_set, tasks_scope_future)
            if (o_h, o_f) == (h_sig, f_sig):
                return False
            suffix += 1

    def _make_hf_basekey(
            self,
            sol: Dict[str, Any],
            tasks_scope_done: List[str],
            tasks_done_set: Set[str],
            tasks_scope_future: List[str]
        ) -> Tuple[
            Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...],
            Tuple[Tuple[str, int, int], ...],
            str,
            Dict[str, int]
        ]:
        sol = self._ensure_compact_solution(sol)
        start_times = self._get_start_times(sol)
        finish_times = self._compute_finish_times(start_times)
        h_sig = make_hist_signature(
            finish_times=finish_times,
            durations=self.project_data.durations,
            resources=self.project_data.resources,
            tasks_in_scope=tasks_scope_done
        )
        f_sig = make_frontier_signature(
            dependencies=sol["dependencies"],
            finish_times=finish_times,
            tasks_done_scope=tasks_done_set,
            tasks_future=tasks_scope_future
        )
        base_key = self._digest_pair_signature(h_sig, f_sig)
        return h_sig, f_sig, base_key, finish_times

    def _hist_key(
            self,
            h_sig: Tuple[Tuple[str, Tuple[Tuple[int, int], ...]], ...]
        ) -> str:
        import hashlib
        # Digest compacto e estável para uso como chave (evita tuplas grandes em RAM)
        payload = repr(h_sig).encode("utf-8")
        return hashlib.blake2s(payload, digest_size=12).hexdigest()

    def _frontier_ctx_make(
            self,
            tasks_done_set: Set[str],
            downstream_tasks: List[str]
        ) -> Dict[str, Any]:
        preds_by_task = {
            t: [p for p in self.project_data.mandatory_graph.get(t, []) if p in tasks_done_set]
            for t in downstream_tasks
        }
        c_by_task = {t: len(preds_by_task[t]) for t in downstream_tasks}
        return {
            "downstream": tuple(downstream_tasks),
            "preds_by_task": preds_by_task,
            "c_by_task": c_by_task
        }

    def _frontier_vec(
            self,
            finish_times: Dict[str, int],
            frontier_ctx: Dict[str, Any]
        ) -> Dict[str, Tuple[int, int]]:
        preds_by_task = frontier_ctx["preds_by_task"]
        c_by_task = frontier_ctx["c_by_task"]
        vec = {}
        for t, preds in preds_by_task.items():
            max_pred_ft = 0
            if preds:
                # calcula apenas o necessário para o "max"
                for p in preds:
                    v = finish_times.get(p, 0)
                    if v > max_pred_ft:
                        max_pred_ft = v
            vec[t] = (c_by_task[t], max_pred_ft)
        return vec

    def _update_frontier_group(
            self,
            group_pairs: List[Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]],
            new_pair: Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]],
            downstream_tasks: List[str]
        ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]], bool, int]:
        """
        Retorna (grupo_atualizado, candidata_ficou, removidos_total),
        onde 'removidos_total' inclui a própria candidata se for dominada.
        """
        new_sol, new_vec = new_pair
        kept = True
        kept_list: List[Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]] = []

        # Se algum existente domina a candidata, ela sai e nada mais muda
        for s, v in group_pairs:
            dominates = True
            better = False
            for t in downstream_tasks:
                c_o, t_o = v[t]; c_n, t_n = new_vec[t]
                if c_o < c_n or t_o > t_n:
                    dominates = False
                    break
                if c_o > c_n or t_o < t_n:
                    better = True
            if dominates and better:
                return group_pairs, False, 1  # candidata é descartada (contabiliza DF)

        # Caso a candidata não seja dominada, removemos os existentes por ela dominados
        for s, v in group_pairs:
            dominated_by_new = True
            better2 = False
            for t in downstream_tasks:
                c_o, t_o = v[t]; c_n, t_n = new_vec[t]
                if c_n < c_o or t_n > t_o:
                    dominated_by_new = False
                    break
                if c_n > c_o or t_n < t_o:
                    better2 = True
            if dominated_by_new and better2:
                continue  # removido
            kept_list.append((s, v))

        kept_list.append((new_sol, new_vec))
        removed_count = len(group_pairs) + 1 - len(kept_list)
        return kept_list, True, removed_count

    def _online_frontier_update(
            self,
            sol: Dict[str, Any],
            frontier_by_hist: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Tuple[int, int]]]]],
            unique_keys: Set[str],
            basekey_by_id: Dict[int, str],
            frontier_ctx: Dict[str, Any],
            tasks_scope_done: List[str],
            tasks_done_set: Set[str],
            tasks_scope_future: List[str]
        ) -> Tuple[int, int, int]:
        aprov_delta = 0
        hf_delta = 0
        df_delta = 0

        # Compactação garantida
        sol = self._ensure_compact_solution(sol)

        # Assinaturas (usa digest leve como chave de histograma)
        h_sig, _, base_key, finish_times = self._make_hf_basekey(
            sol, tasks_scope_done, tasks_done_set, tasks_scope_future
        )
        h_key = self._hist_key(h_sig)

        # HF: deduplicação por base_key em Set[str]
        if self.config.ENABLE_HF_PRUNING and base_key in unique_keys:
            return 0, 1, 0
        if self.config.ENABLE_HF_PRUNING:
            unique_keys.add(base_key)
            basekey_by_id[id(sol)] = base_key

        # DF: manutenção incremental do fronte por histograma, com vetores pré-computados
        cand_vec = self._frontier_vec(finish_times, frontier_ctx)
        group_pairs = frontier_by_hist.get(h_key)
        if group_pairs is None:
            frontier_by_hist[h_key] = [(sol, cand_vec)]
            return 1, 0, 0

        updated_pairs, kept_new, removed_count = self._update_frontier_group(
            group_pairs, (sol, cand_vec), list(frontier_ctx["downstream"])
        )
        df_delta += removed_count

        # Remover chaves HF dos que saíram do fronte (inclui a candidata se dominada)
        old_ids = set(id(s) for s, _ in group_pairs) | {id(sol)}
        new_ids = set(id(s) for s, _ in updated_pairs)
        removed_ids = old_ids - new_ids
        if removed_ids:
            for rid in removed_ids:
                bk = basekey_by_id.pop(rid, None)
                if bk is not None:
                    unique_keys.discard(bk)

        # Delta de aprovados = variação do tamanho do grupo
        aprov_delta = len(updated_pairs) - len(group_pairs)
        frontier_by_hist[h_key] = updated_pairs
        return aprov_delta, hf_delta, df_delta

    def _iter_parent_results_seq(self, parent_solutions, layer_task_options, tasks_scope_done):
        worker = Worker(self.project_data, self.config, layer_task_options, tasks_scope_done)
        for p in parent_solutions:
            yield worker.run(p)

    def _iter_parent_results_mp(self, parent_solutions, layer_task_options, tasks_scope_done, procs, chunks, min_parents):
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(
            processes=procs,
            initializer=MPWorkerBridge.init_child,
            initargs=(self.project_data, self.config, layer_task_options, tasks_scope_done)
        ) as pool:
            it = pool.imap_unordered(MPWorkerBridge.run_parent, parent_solutions, chunksize=chunks)
            for res in it:
                yield res

    def _compute_finish_times(self, start_times: Dict[str, int]) -> Dict[str, int]:
        """
        Calcula tempos de término diretamente, sem indexação vetorial.
        Mantém resultados idênticos ao método anterior.
        """
        durs = self.project_data.durations
        return {tid: int(start_times.get(tid, 0)) + int(durs[tid]) for tid in self.project_data.task_ids}

    def _digest_pair_signature(self, hist_sig, front_sig) -> str:
        """
        Retorna um digest compacto (hex) para (hist_sig, front_sig).
        Utiliza repr() para serialização determinística e BLAKE2b(16B).
        """ 
        h = hashlib.blake2b(digest_size=16)
        # repr é determinístico para tuplas/strings/ints já ordenadas nas funções de assinatura
        h.update(repr((hist_sig, front_sig)).encode("utf-8"))
        return h.hexdigest()

    def _log_layer_statistics(self, layer_num: int, stats: Dict[str, int]) -> None:
            main_logger.log(f"\n  Estatísticas da Camada {layer_num}:")
            main_logger.log(f"    - Cronogramas Testados: {stats.get('cronogramas_testados', 0):,}")
            main_logger.log(f"    - Descartados por Ciclo: {stats.get('descartados_por_ciclo', 0):,}")
            main_logger.log(f"    - Descartados por Duração: {stats.get('descartados_por_duracao', 0):,}")
            main_logger.log(f"    - Descartados por Gap Futuro: {stats.get('descartados_por_gap_futuro', 0):,}")
            main_logger.log(f"    - Descartados por Vales: {stats.get('descartados_por_vales', 0):,}")
            main_logger.log(f"    - Descartados por Histograma + Fronteira: {stats.get('descartados_por_equivalencia_hist_fronteira', 0):,}")
            main_logger.log(f"    - Descartados por Dominância de Fronteira: {stats.get('descartados_por_dominancia_de_fronteira', 0):,}")

    def _save_intermediate_results(self, solutions: List[Dict[str, Any]], layer_num: int):
        if not solutions: return
        output_dir = os.path.dirname(self.config.OUTPUT_FILE)
        input_basename = self.config._input_basename
        output_path = os.path.join(output_dir, f"{input_basename}_camada_{layer_num}.txt")
        main_logger.log(f"  Salvando {len(solutions):,} resultados intermediários em: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sol in solutions:
                entry = self._format_solution_for_output(sol)
                f.write(str([sol['id']] + entry) + '\n')

    def save_final_results(self, start_time: float) -> None:
        main_logger.log("\n--- ETAPA FINAL: Formatando Resultados ---")
        ts = getattr(self, "total_stats", {}) or {}
        tt = int(ts.get('cronogramas_testados', 0))
        tc = int(ts.get('descartados_por_ciclo', 0))
        td = int(ts.get('descartados_por_duracao', 0))
        tgf = int(ts.get('descartados_por_gap_futuro', 0))
        tv = int(ts.get('descartados_por_vales', 0))
        te = int(ts.get('descartados_por_equivalencia_hist_fronteira', 0))
        tdf = int(ts.get('descartados_por_dominancia_de_fronteira', 0))
        main_logger.log(f"Total de Cronogramas Testados: {tt:,}")
        main_logger.log(f"Total Descartados por Ciclo: {tc:,}")
        main_logger.log(f"Total Descartados por Duração: {td:,}")
        main_logger.log(f"Total Descartados por Gap Futuro: {tgf:,}")
        main_logger.log(f"Total Descartados por Vales: {tv:,}")
        main_logger.log(f"Total Descartados por Histograma + Fronteira: {te:,}")
        main_logger.log(f"Total Descartados por Dominância de Fronteira: {tdf:,}")
        if not getattr(self, "solutions", None):
            main_logger.log("\nNenhuma solução encontrada para salvar.")
            MemUtil.log_summary(main_logger)
            main_logger.log(f"Tempo total de processamento: {time.perf_counter() - start_time:.2f}s")
            return
        all_tasks = list(self.project_data.task_ids)
        durs = self.project_data.durations
        uniq = {}
        for sol in self.solutions:
            st = self._get_start_times(sol)
            ft = {tid: st.get(tid, 0) + durs[tid] for tid in st}
            h_sig = make_hist_signature(ft, self.project_data.durations, self.project_data.resources, all_tasks)
            k = self._digest_pair_signature(h_sig, tuple(sorted((tid, ft.get(tid, 0), durs[tid]) for tid in all_tasks)))
            suf = 1
            while k in uniq and uniq[k][0] == h_sig:
                suf += 1
                k = f"{k}#{suf}"
            if k in uniq:
                suf = 1
                while f"{k}#{suf}" in uniq:
                    suf += 1
                uniq[f"{k}#{suf}"] = (h_sig, sol)
            elif k not in uniq:
                uniq[k] = (h_sig, sol)
        final_list = [v[1] for v in uniq.values()]
        main_logger.log(f"\nResultados de Cronogramas válidos: {len(final_list):,} ")
        main_logger.log(f"Tempo total de processamento: {time.perf_counter() - start_time:.2f}s")
        MemUtil.log_summary(main_logger)
        main_logger.log(f"\nSalvos em: {self.config.OUTPUT_FILE}")
        with open(self.config.OUTPUT_FILE, 'w', encoding='utf-8') as f:
            for sol in sorted(final_list, key=lambda x: str(x.get('id'))):
                entry = self._format_solution_for_output(sol)
                f.write(str([sol.get('id')] + entry) + '\n')

    def save_layer_0_comparison_files(self):
        main_logger.log("\n--- ETAPA ADICIONAL: Gerando Comparativo da Camada 0 ---")
        if not self.layer_0_solutions:
            main_logger.log("  Nenhuma solução da camada 0 foi gerada. Arquivos de comparativo não serão criados.")
            return
        successful_root_ids = {sol['id'].split('.')[0] for sol in self.solutions}
        output_dir = os.path.dirname(self.config.OUTPUT_FILE)
        input_basename = self.config._input_basename
        success_file_path = os.path.join(output_dir, f"{input_basename}_camada_0_s.txt")
        failure_file_path = os.path.join(output_dir, f"{input_basename}_camada_0_f.txt")
        success_schedules, failure_schedules = [], []
        for sol in self.layer_0_solutions:
            if sol['id'] in successful_root_ids: success_schedules.append(sol)
            else: failure_schedules.append(sol)
        with open(success_file_path, 'w', encoding='utf-8') as f:
            for sol in sorted(success_schedules, key=lambda x: str(x['id'])):
                entry = self._format_solution_for_output(sol)
                f.write(str([sol['id']] + entry) + '\n')
        main_logger.log(f"  Salvos {len(success_schedules)} cronogramas da Camada 0 que TIVERAM SUCESSO em: {success_file_path}")
        with open(failure_file_path, 'w', encoding='utf-8') as f:
            for sol in sorted(failure_schedules, key=lambda x: str(x['id'])):
                entry = self._format_solution_for_output(sol)
                f.write(str([sol['id']] + entry) + '\n')
        main_logger.log(f"  Salvos {len(failure_schedules)} cronogramas da Camada 0 que FALHARAM em: {failure_file_path}")

    def _format_solution_for_output(self, solution: Dict[str, Any]) -> List[Tuple[str, int, str, str]]:
        deps = solution.get('dependencies', {})
        entry = []
        for task_info in self.project_data.all_tasks_info:
            task_id = task_info['id']
            mand_preds = set(self.project_data.mandatory_graph.get(task_id, []))
            all_preds = set(deps.get(task_id, []))
            arb_preds = sorted(list(all_preds - mand_preds))
            entry.append((task_id, task_info['duration'], ','.join(sorted(list(mand_preds))), ','.join(arb_preds)))
        return entry

# ==================================================================================================
# --- 5. WORKER PARA MULTIPROCESSAMENTO ---
# ==================================================================================================
class Worker:
    def __init__(self, project_data, config, layer_task_options, tasks_scope_for_checks):
        self.project_data = project_data
        self.config = config
        self.layer_task_options = layer_task_options
        self.tasks_scope_for_checks = list(tasks_scope_for_checks) if tasks_scope_for_checks else list(project_data.task_ids)
        self._es_cache = OrderedDict()

    def _build_ancestor_closure(self, deps: dict, task_ids: list) -> dict:
        """
        Retorna {tarefa -> conjunto de ancestrais} a partir de 'deps'.
        Usado para filtrar predecessores arbitrários redundantes.
        """
        adj = {t: list(deps.get(t, [])) for t in task_ids}

        @lru_cache(maxsize=None)
        def dfs(t: str) -> frozenset:
            acc = set()
            for p in adj.get(t, ()):
                if p not in acc:
                    acc.add(p)
                    acc.update(dfs(p))
            return frozenset(acc)

        return {t: set(dfs(t)) for t in task_ids}

    def sanitize_options(self, parent_deps_base):
        """
        Igual às regras atuais de sanitização, mas com ordenação "block-first".
        """
        task_ids = list(self.project_data.task_ids)
        closure = self._build_ancestor_closure(parent_deps_base, task_ids)
        sanitized: Dict[str, List[Tuple[str, ...]]] = {}

        # 1) Sanitização padrão (idêntica)
        for task_id, options in self.layer_task_options.items():
            mandatory = set(parent_deps_base.get(task_id, ()))
            seen, new_opts = set(), []
            for opt in (options or [tuple()]):
                filtered = [
                    p for p in opt
                    if p != task_id and p not in mandatory and p in task_ids
                    and p not in closure.get(task_id, set())
                ]
                canon = tuple(sorted(set(filtered)))
                if canon not in seen:
                    seen.add(canon); new_opts.append(canon)
            if not new_opts:
                new_opts = [tuple()]
            sanitized[task_id] = new_opts

        # 2) Ordenação “block-first” das tarefas
        bw = getattr(self.project_data, "block_weight", {})
        tb = getattr(self.project_data, "task_to_block", {})
        nr = getattr(self.project_data, "task_to_network_duration", {})

        def task_key(tid: str) -> Tuple[int, int, int, int, str]:
            bid = tb.get(tid)
            w = bw.get(bid, self.project_data.durations.get(tid, 0))
            k_opts = len(sanitized.get(tid, []))
            return (-int(w), -int(nr.get(tid, 0)), -k_opts, 0, tid)

        tasks_in_layer = list(sanitized.keys())
        tasks_with_options = sorted([tid for tid in tasks_in_layer if sanitized.get(tid)], key=task_key)

        # 3) Reordena as opções preferindo predecessores de blocos pesados
        def combo_score(opt: Tuple[str, ...]) -> Tuple[int, int, int]:
            if not opt:
                return (10**9, 1, 0)  # vazio por último
            total = 0
            for p in opt:
                pbid = tb.get(p)
                total += bw.get(pbid, self.project_data.durations.get(p, 0))
            # mais pesado primeiro; depois menos predecessores; depois ordem lexicográfica
            return (-int(total), len(opt), 0)

        option_lists: List[List[Tuple[str, ...]]] = []
        for tid in tasks_with_options:
            opts = list(sanitized[tid])
            opts.sort(key=combo_score)
            option_lists.append(opts)

        return tasks_with_options, option_lists

    def _build_dependencies_with_choice(self, parent_deps, tasks, choice):
        """
        Constrói o grafo de predecessores a partir do pai + escolha de arbitrários.
        """
        deps = {t: list(preds) for t, preds in parent_deps.items()}
        for task_id, arb_preds in zip(tasks, choice):
            if arb_preds:
                base = deps.get(task_id, [])
                base_set = set(base)
                for p in arb_preds:
                    if p not in base_set:
                        base.append(p)
                        base_set.add(p)
                deps[task_id] = base
            else:
                deps.setdefault(task_id, list(deps.get(task_id, [])))
        for t in self.project_data.task_ids:
            deps.setdefault(t, deps.get(t, []))
        return deps    

    def _deps_signature(self, deps):
        """
        Assinatura imutável do grafo de predecessores (para cache ES/EF).
        """
        items = []
        for t in sorted(self.project_data.task_ids):
            preds = tuple(sorted(deps.get(t, [])))
            items.append((t, preds))
        return tuple(items)

    def _start_times_from_sig(self, sig):
        if sig in self._es_cache:
            val = self._es_cache[sig]
            self._es_cache.move_to_end(sig)
            return val
        pred_graph = {t: list(preds) for t, preds in sig}
        st = calculate_earliest_start_times(pred_graph, self.project_data.durations, self.project_data.task_ids)
        if st is None:
            self._es_cache[sig] = None
            self._es_cache.move_to_end(sig)
            return None
        st_tuple = tuple(sorted(st.items()))
        self._es_cache[sig] = st_tuple
        self._es_cache.move_to_end(sig)
        return st_tuple

    def evaluate_choice(self, parent_id, parent_deps_base, tasks_with_options, choice):
        stats = defaultdict(int)
        current_deps = self._build_dependencies_with_choice(parent_deps_base, tasks_with_options, choice)
        sig = self._deps_signature(current_deps)
        st_tuple = self._start_times_from_sig(sig)
        if st_tuple is None:
            stats["descartados_por_ciclo"] += 1
            return None, stats
        start_times = dict(st_tuple)
        durs = self.project_data.durations
        max_finish = 0
        for tid, st in start_times.items():
            ft = int(st) + int(durs[tid])
            if ft > max_finish:
                max_finish = ft
        if max_finish > self.project_data.max_duration_allowed:
            stats["descartados_por_duracao"] += 1
            return None, stats
        threshold = int(getattr(self.config, "ENABLE_FUTURE_GAP_PRUNING", 0) or 0)
        if threshold > 0:
            info = getattr(self.project_data, "prunable_resources_info", None)
            if info:
                for res_id, res_info in info.items():
                    early = res_info.get("early_tasks", [])
                    late = res_info.get("late_tasks", [])
                    if not early or not late:
                        continue
                    for et in early:
                        for lt in late:
                            if et in start_times and lt in start_times and start_times[lt] < start_times[et]:
                                stats["descartados_por_gap_futuro"] += 1
                                return None, stats
        scope = self.tasks_scope_for_checks
        if getattr(self.config, "MAX_ALLOWED_UP_VALLEYS", None) is not None:
            limit = int(self.config.MAX_ALLOWED_UP_VALLEYS)
            total_valleys = calculate_resource_valleys(start_times, durs, self.project_data.resources, scope, limit=limit)
            if total_valleys > limit:
                stats["descartados_por_vales"] += 1
                return None, stats
        solution = {"id": None, "dependencies": current_deps, "start_times": start_times}
        return solution, stats

    def run(self, parent_solution: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        solutions_found: List[Dict[str, Any]] = []
        stats: Dict[str, int] = defaultdict(int)

        parent_id = parent_solution["id"]
        parent_deps_base = parent_solution["dependencies"]

        tasks_with_options, option_lists = self.sanitize_options(parent_deps_base)
        if not tasks_with_options:
            return [parent_solution], stats

        sub_id_counter = 0
        for choice in product(*option_lists):
            stats["cronogramas_testados"] += 1

            sol, delta = self.evaluate_choice(parent_id, parent_deps_base, tasks_with_options, choice)
            for k, v in delta.items():
                stats[k] += v
            if sol is None:
                continue

            sub_id_counter += 1
            sol["id"] = f"{parent_id}.{sub_id_counter}"
            solutions_found.append(sol)

        return solutions_found, stats

class MPWorkerBridge:
    """
    Ponte para multiprocessamento: inicializa um Worker por processo-filho
    e executa a avaliação de um estado-pai.
    """
    _instance: Optional["Worker"] = None

    @classmethod
    def init_child(cls, project_data, config, layer_task_options, tasks_scope_for_checks) -> None:
        cls._instance = Worker(
            project_data=project_data,
            config=config,
            layer_task_options=layer_task_options,
            tasks_scope_for_checks=tasks_scope_for_checks
        )

    @classmethod
    def run_parent(cls, parent_solution: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        if cls._instance is None:
            raise RuntimeError("MPWorkerBridge não inicializado. Verifique o initializer do Pool.")
        return cls._instance.run(parent_solution)

# ==================================================================================================
# --- 6. FUNÇÃO PRINCIPAL ---
# ==================================================================================================
class App:
    @staticmethod
    def run() -> None:
        start_time = time.perf_counter()
        
        MemUtil.enable(sample_period_s=3.0)        
        
        try:
            config = Config()
            project_data = ProjectData(config)
            scheduler = Scheduler(project_data, config)
            scheduler.run_optimization()
            scheduler.save_final_results(start_time)
            scheduler.save_layer_0_comparison_files()
        except (ValueError, FileNotFoundError) as e:
            main_logger.log(f"\nERRO: {e}")
        except Exception as e:
            main_logger.log(f"\nOcorreu um erro inesperado: {e}")
            traceback.print_exc()
        finally:
            main_logger.log(f"\n--- FIM DA EXECUÇÃO ---")
            print(f"Execução finalizada. Log completo salvo em: {main_logger.log_file_path}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    App.run()