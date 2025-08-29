
from __future__ import annotations
import ast
import os
import sys
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Iterable


class PodaPorHistograma:
    """
    Deduplicador por histograma de recursos.
    Lê:
      - recursos: arquivo com lista de tuplas (Tarefa, Duracao, PredObrig, PredArb, Recurso, Qtd)
      - cronogramas: uma linha por cronograma, cada linha é uma lista [id, (T, dur, obrig, arb), ...]
    Gera:
      - arquivo de saída com apenas cronogramas únicos segundo a assinatura de histograma.
    """
    def __init__(self, recursos_path: str, cronogramas_path: str, saida_path: str) -> None:
        self.recursos_path = recursos_path
        self.cronogramas_path = cronogramas_path
        self.saida_path = saida_path
        self.task_duration: Dict[str, int] = {}
        self.task_resource_id: Dict[str, str] = {}
        self.task_resource_qty: Dict[str, int] = {}
        self.resource_ids: List[str] = []

    def carregar_recursos(self) -> None:
        """
        Carrega o arquivo de recursos (lista de tuplas) e popula:
          - task_duration, task_resource_id, task_resource_qty, resource_ids (ordenada)
        O parser ignora cabeçalhos/texto e extrai o primeiro bloco [...].
        """
        with open(self.recursos_path, "r", encoding="utf-8") as f:
            txt = f.read()
        start = txt.find("[")
        end = txt.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Não foi possível encontrar uma lista de tuplas em recursos.")
        tuples_list = ast.literal_eval(txt[start:end + 1])
        resources_set = set()
        for row in tuples_list:
            tarefa, dur, _po, _pa, rid, qty = row
            self.task_duration[tarefa] = int(dur)
            self.task_resource_id[tarefa] = str(rid)
            self.task_resource_qty[tarefa] = int(qty)
            resources_set.add(str(rid))
        self.resource_ids = sorted(resources_set)

    def _parse_linha_cronograma(self, line: str) -> Tuple[str, List[Tuple[str, str, str]]]:
        """
        Converte uma linha do arquivo de cronogramas em:
          - id do cronograma
          - lista de tuplas: (tarefa, pred_obrig, pred_arb)
        Ignora campos de duração da linha (usa duração da base de recursos).
        """
        if not line.strip():
            raise ValueError("Linha vazia.")
        data = ast.literal_eval(line)
        if not isinstance(data, list) or not data:
            raise ValueError("Formato inesperado de linha de cronograma.")
        cron_id = str(data[0])
        itens = []
        for t in data[1:]:
            tarefa = t[0]
            pred_obrig = t[2] if len(t) > 2 else ""
            pred_arb = t[3] if len(t) > 3 else ""
            itens.append((tarefa, pred_obrig or "", pred_arb or ""))
        return cron_id, itens

    def _construir_grafo(self, itens: Iterable[Tuple[str, str, str]]) -> Dict[str, List[str]]:
        """
        A partir de (tarefa, pred_obrig, pred_arb) cria lista de predecessores por tarefa.
        Preditores vazios são ignorados. Se o predecessor não existir na base, é ignorado.
        """
        preds: Dict[str, List[str]] = defaultdict(list)
        tarefas = {t for t, _po, _pa in itens}
        for t, po, pa in itens:
            for p in (po, pa):
                if p and p in tarefas:
                    preds[t].append(p)
            if t not in preds:
                preds[t] = preds.get(t, [])
        return preds

    def _topos_early_starts(self, preds: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Calcula início mais cedo (earliest start) por Kahn, sem nivelamento de recursos.
        start[t] = max(finish[p] para p in preds[t]); finish[t] = start[t] + dur[t].
        """
        indeg: Dict[str, int] = {t: 0 for t in preds}
        for t in preds:
            for p in preds[t]:
                indeg[t] += 1
        q = deque([t for t, d in indeg.items() if d == 0])
        start: Dict[str, int] = {t: 0 for t in preds}
        processed = 0
        children: Dict[str, List[str]] = defaultdict(list)
        for t in preds:
            for p in preds[t]:
                children[p].append(t)
        while q:
            u = q.popleft()
            processed += 1
            finish_u = start[u] + self.task_duration[u]
            for v in children.get(u, []):
                start[v] = max(start[v], finish_u)
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if processed != len(preds):
            raise ValueError("Ciclo detectado no grafo de precedência.")
        return start

    def _uso_por_recurso(self, starts: Dict[str, int]) -> Dict[str, List[int]]:
        """
        Constrói a linha do tempo densa de uso para cada recurso (unidade = dia).
        Para cada tarefa t ativa no intervalo [start, finish), soma a quantidade do seu recurso.
        """
        makespan = 0
        for t, s in starts.items():
            makespan = max(makespan, s + self.task_duration[t])
        timeline: Dict[str, List[int]] = {rid: [0] * makespan for rid in self.resource_ids}
        for t, s in starts.items():
            rid = self.task_resource_id[t]
            qty = self.task_resource_qty[t]
            f = s + self.task_duration[t]
            for d in range(s, f):
                timeline[rid][d] += qty
        return timeline

    def _assinatura_histograma(self, timeline: Dict[str, List[int]]) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        """
        Assinatura canônica: tupla ordenada por recurso (rid, (valores...)).
        """
        sig: List[Tuple[str, Tuple[int, ...]]] = []
        for rid in self.resource_ids:
            sig.append((rid, tuple(timeline[rid])))
        return tuple(sig)

    @staticmethod
    def caminhos_padrao(base_dir: str) -> Tuple[str, str, str]:
        """
        Monta caminhos padrão para recursos, cronogramas e saída dentro de base_dir.
        """
        recursos = os.path.join(base_dir, "R22_tarefas_4.txt")
        cronos = os.path.join(base_dir, "R22_tarefas_4_resultados_unicos.txt")
        saida = os.path.join(base_dir, "R22_tarefas_4_resultados_unicos_hist_dedup.txt")
        return recursos, cronos, saida

    def processar(self) -> Tuple[int, int, str]:
        """
        Lê todos os cronogramas, gera assinatura de histograma e deduplica.
        Retorna (total_linhas, unicos, caminho_saida).
        """
        self.carregar_recursos()
        unicos_por_sig: Dict[Tuple[Tuple[str, Tuple[int, ...]], ...], str] = {}
        linhas_unicas: List[str] = []
        total = 0
        with open(self.cronogramas_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    _cid, itens = self._parse_linha_cronograma(line)
                    preds = self._construir_grafo(itens)
                    starts = self._topos_early_starts(preds)
                    timeline = self._uso_por_recurso(starts)
                    sig = self._assinatura_histograma(timeline)
                    if sig not in unicos_por_sig:
                        unicos_por_sig[sig] = line
                        linhas_unicas.append(line)
                except Exception:
                    continue
        with open(self.saida_path, "w", encoding="utf-8") as f:
            for ln in linhas_unicas:
                f.write(ln + "\n")
        return total, len(linhas_unicas), os.path.abspath(self.saida_path)


if __name__ == "__main__":
    base_dir_default = r"C:\Users\farmc\Downloads\res_automacao"
    if len(sys.argv) >= 4:
        recursos_path = sys.argv[1]
        cronos_path = sys.argv[2]
        saida_path = sys.argv[3]
    else:
        recursos_path, cronos_path, saida_path = PodaPorHistograma.caminhos_padrao(base_dir_default)
    app = PodaPorHistograma(recursos_path, cronos_path, saida_path)
    total, unicos, path = app.processar()
    print(f"Total lidos: {total} | Únicos por histograma: {unicos}")
    print(f"Arquivo salvo em: {path}")
