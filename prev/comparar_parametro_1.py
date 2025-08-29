# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import re
import ast

class Config:
    """
    Configurações de diretórios e arquivos de entrada.
    Ajuste DIR e nomes, se necessário.
    """
    DIR = Path(r"C:\Users\farmc\Downloads\res_automacao")

    ARQ_A_PARAM = DIR / "R22_tarefas_4_parametros.txt"
    ARQ_B_PARAM = DIR / "R31_tarefas_4_parametros.txt"

    ARQ_A_UNICOS = DIR / "R22_tarefas_4_resultados_unicos.txt"
    ARQ_B_UNICOS = DIR / "R31_tarefas_4_resultados_unicos.txt"

    PARAMS = [
        "duracao", "folga", "linearidade", "expansao", "regularidade",
        "ele_max", "est_max", "mec_max", "tub_max",
        "ele_%", "est_%", "mec_%", "tub_%"
    ]

class FileIO:
    """Funções utilitárias de entrada/saída de dados."""

    @staticmethod
    def ler_parametros_csv(caminho: Path) -> pd.DataFrame:
        """
        Lê CSV/TSV/semicolumn detectando separador automaticamente (engine='python').
        Tenta UTF-8 e recorre a Latin-1 em caso de erro.
        Requer coluna 'id'.
        """
        try:
            df = pd.read_csv(caminho, sep=None, engine="python", encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(caminho, sep=None, engine="python", encoding="latin-1")
        if "id" not in df.columns:
            raise KeyError(f"Coluna 'id' ausente em {caminho.name}.")
        return df

    @staticmethod
    def ler_linhas_texto(caminho: Path) -> list[str]:
        """
        Lê o arquivo como texto bruto, preservando uma linha por cronograma.
        Útil para '..._resultados_unicos.txt'.
        """
        try:
            txt = caminho.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            txt = caminho.read_text(encoding="latin-1")
        linhas = [ln.strip() for ln in txt.splitlines() if ln.strip()]
        return linhas

    @staticmethod
    def escrever_texto(caminho: Path, conteudo: str) -> None:
        """Escreve texto em UTF-8, garantindo a criação de pastas."""
        caminho.parent.mkdir(parents=True, exist_ok=True)
        caminho.write_text(conteudo, encoding="utf-8")

class NameUtil:
    """Utilidades para extrair identificadores de nomes de arquivo."""

    @staticmethod
    def extrair_id_compacto(nome: str) -> str:
        """
        Extrai 'R\\d+' do nome (ex.: 'R22'), para compor a pasta 'diferencas_parametros_R22_R28'.
        Fallback: usa o stem até o primeiro '_' se não houver padrão.
        """
        m = re.search(r"(R\d+)", nome)
        if m:
            return m.group(1)
        base = Path(nome).stem
        return base.split("_")[0]

    @staticmethod
    def base_sem_sufixo_parametro(nome: str) -> str:
        """
        'R22_tarefas_4_parametros.txt' -> 'R22_tarefas_4'
        Remove sufixos '_parametros'/'_param'/'_parameters' (case-insensitive).
        """
        stem = Path(nome).stem
        return re.sub(r"(_param(et)?ros?|_parameters?)$", "", stem, flags=re.IGNORECASE)

class IdUtil:
    """
    Normalização de IDs para casamento entre parâmetros e resultados_unicos.
    - Remove espaços;
    - Se for numérico (somente dígitos), remove zeros à esquerda (ex.: '066' -> '66').
    - Caso contrário, usa lowercase para reduzir variações de caixa.
    """
    @staticmethod
    def normalizar_id(x: object) -> str:
        s = str(x).strip()
        return str(int(s)) if s.isdigit() else s.lower()
    
    @staticmethod
    def canon_param_id(x: object) -> str:
        """
        Canoniza o ID vindo do ARQUIVO DE PARÂMETROS para buscar no resultados_unicos:
        - Remove espaços;
        - Se parecer número (ex.: '66' ou '66.0'), usa inteiro em string: '66';
        - Caso contrário, retorna a string limpa.
        (Obs.: serve apenas para casamento interno entre 'parâmetros' e 'resultados_unicos'.)
        """
        s = str(x).strip()
        if s == "":
            return s
        s_num = s.replace(",", ".")
        try:
            val = float(s_num)
            if val.is_integer():
                return str(int(val))
            return f"{val:.6f}".rstrip("0").rstrip(".")
        except Exception:
            return s

class CronogramaUtil:
    """Formatação e detecção de coluna de cronograma."""

    @staticmethod
    def detectar_coluna_cronograma(df: pd.DataFrame) -> str | None:
        """
        Preferência por nomes usuais; senão, escolhe a última coluna do tipo 'object'.
        """
        candidatos = ["cronograma", "Cronograma", "CRONOGRAMA", "cron", "schedule"]
        for c in candidatos:
            if c in df.columns:
                return c
        obj_cols = [c for c in df.columns if str(df[c].dtype) == "object"]
        return obj_cols[-1] if obj_cols else None

    @staticmethod
    def formatar_linha(pid: str, cron_str: str) -> str:
        """
        Gera: ['<ID>', ...tarefas...]
        Normaliza para não duplicar colchetes/aspas.
        """
        s = str(cron_str).strip()
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            s = s[1:-1].strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
        return f"['{str(pid)}', {s}]" if s else f"['{str(pid)}']"

class ValueUtil:
    """Normalização e ordenação de valores de parâmetros."""

    @staticmethod
    def normalizar_valor(x) -> str | None:
        """
        Converte para número quando possível; arredonda floats a 6 casas; devolve string.
        """
        try:
            v = pd.to_numeric(x)
        except Exception:
            return None
        if pd.isna(v):
            return None
        if isinstance(v, float):
            v = round(v, 6)
        if float(v).is_integer():
            return str(int(v))
        return f"{float(v):.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def ordenar_safely(valores: set[str]) -> list[str]:
        """Ordena como float quando possível; senão, lexicográfico."""
        try:
            return sorted(valores, key=lambda x: float(x))
        except Exception:
            return sorted(valores)

class CronogramaParser:
    """
    Faz o parse de uma linha no formato:
    ['<ID>', ('T1', 3, '', ''), ('T2', 3, '', ''), ...]
    Retorna (id_str, itens_str) onde itens_str é "('T1', 3, '', ''), ('T2', ...)".
    """

    @staticmethod
    def _items_to_str(items: list) -> str:
        """
        Converte a lista dos itens do cronograma para string Pythonic,
        preservando tuplas/listas internas (ex.: "('T1', 3, '', ''), ('T2', 2, '', '')").
        """
        if items is None or len(items) == 0:
            return ""
        return ", ".join(str(it) for it in items)

    @staticmethod
    def parse_linha(linha: str) -> tuple[str | None, str]:
        """
        Tenta via ast.literal_eval; se falhar, aplica fallback por split do primeiro elemento.
        Retorna (id_str, itens_str). Se não conseguir extrair ID, retorna (None, "").
        """
        s = linha.strip()
        # Caminho principal: literal_eval
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)) and len(obj) >= 1:
                pid = obj[0]
                itens = list(obj[1:]) if len(obj) > 1 else []
                return str(pid), CronogramaParser._items_to_str(itens)
        except Exception:
            pass

        # Fallback: split do primeiro elemento fora de aspas
        if not (s.startswith("[") and s.endswith("]")):
            return None, ""
        inner = s[1:-1].strip()
        if not inner:
            return None, ""

        id_part, rest = CronogramaParser._split_primeiro_elemento(inner)
        pid = CronogramaParser._limpar_id(id_part)
        return (pid if pid != "" else None), rest.strip()

    @staticmethod
    def _split_primeiro_elemento(inner: str) -> tuple[str, str]:
        """
        Divide 'inner' em (primeiro_elemento, resto) respeitando aspas e parênteses.
        Ex.: "'123', ('T1',3)" -> ("'123'", "('T1',3)")
        """
        i, n = 0, len(inner)
        quote = None
        depth_par = depth_col = depth_ch = 0  # (), [], {}
        while i < n:
            ch = inner[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if ch == quote:
                    quote = None
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                i += 1
                continue
            if ch == "(":
                depth_par += 1
            elif ch == ")":
                depth_par -= 1
            elif ch == "[":
                depth_col += 1
            elif ch == "]":
                depth_col -= 1
            elif ch == "{":
                depth_ch += 1
            elif ch == "}":
                depth_ch -= 1
            elif ch == "," and depth_par == depth_col == depth_ch == 0:
                # vírgula separadora do primeiro elemento
                primeiro = inner[:i].strip()
                resto = inner[i+1:].strip()
                return primeiro, resto
            i += 1
        # não encontrou vírgula: só um elemento
        return inner.strip(), ""

    @staticmethod
    def _limpar_id(id_part: str) -> str:
        """Remove aspas simples/duplas do ID, se existirem."""
        t = id_part.strip()
        if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
            return t[1:-1].strip()
        return t

class ComparadorParametros:
    """
    Compara valores de parâmetros entre dois arquivos de parâmetros (A e B),
    gera relatório de diferenças e copia cronogramas correspondentes a cada diferença.
    """

    def __init__(self,
                 arq_a_param: Path, arq_b_param: Path,
                 arq_a_unicos: Path, arq_b_unicos: Path,
                 params: list[str], id_col: str = "id"):
        self.arq_a_param = arq_a_param
        self.arq_b_param = arq_b_param
        self.arq_a_unicos = arq_a_unicos
        self.arq_b_unicos = arq_b_unicos
        self.params = params
        self.id_col = id_col

        # Carregar parâmetros (com coluna 'id')
        self.df_a = FileIO.ler_parametros_csv(arq_a_param)
        self.df_b = FileIO.ler_parametros_csv(arq_b_param)

        # Carregar cronogramas como linhas de texto e parsear para DF(id, itens_str)
        self.df_a_un = self._carregar_cronogramas(arq_a_unicos)
        self.df_b_un = self._carregar_cronogramas(arq_b_unicos)

        # Identificadores e pastas de saída
        self.id_a = NameUtil.extrair_id_compacto(arq_a_param.name)
        self.id_b = NameUtil.extrair_id_compacto(arq_b_param.name)
        self.base_rel = f"diferencas_parametros_{self.id_a}_{self.id_b}"
        self.dir_out = (arq_a_param.parent / self.base_rel)
        self.dir_out.mkdir(parents=True, exist_ok=True)

        # Bases para nomes dos arquivos por parâmetro
        self.base_a = NameUtil.base_sem_sufixo_parametro(arq_a_param.name)
        self.base_b = NameUtil.base_sem_sufixo_parametro(arq_b_param.name)

    def _carregar_cronogramas(self, caminho: Path) -> pd.DataFrame:
        """
        Constrói DataFrame com colunas:
        - 'id'       : ID original (string) extraído da linha
        - 'id_norm'  : ID normalizado (para casamento: remove zeros à esquerda se numérico)
        - 'raw_line' : linha bruta do cronograma exatamente como no arquivo
        A partir de um arquivo de texto com UMA LINHA por cronograma.
        """
        linhas = FileIO.ler_linhas_texto(caminho)
        ids, ids_norm, raw = [], [], []
        for ln in linhas:
            pid, _rest = CronogramaParser.parse_linha(ln)
            if pid is None:
                continue
            ids.append(str(pid))
            ids_norm.append(IdUtil.normalizar_id(pid))
            raw.append(ln)  # guarda a linha exatamente como está no arquivo

        df = pd.DataFrame({"id": ids, "id_norm": ids_norm, "raw_line": raw})
        # Remove duplicidades pelo ID normalizado (mantém a primeira ocorrência)
        df = df.drop_duplicates(subset=["id_norm"], keep="first").reset_index(drop=True)
        return df

    def _valores_unicos_normalizados(self, df: pd.DataFrame, coluna: str) -> set[str]:
        """Valores únicos normalizados de uma coluna."""
        if coluna not in df.columns:
            return set()
        s = df[coluna].map(ValueUtil.normalizar_valor)
        return {v for v in s if v is not None}

    def _ids_por_valores(self, df: pd.DataFrame, param: str, exclusivos: set[str]) -> set[str]:
        """
        Retorna IDs cujo valor do parâmetro pertence a 'exclusivos' (sempre coluna 'id').
        """
        if not exclusivos or param not in df.columns or self.id_col not in df.columns:
            return set()
        tmp = df[[self.id_col]].copy()
        tmp["__val"] = df[param].map(ValueUtil.normalizar_valor)
        tmp = tmp[tmp["__val"].isin(exclusivos)].dropna(subset=["__val"])
        tmp = tmp.drop_duplicates(subset=[self.id_col, "__val"])
        return set(map(str, tmp[self.id_col].astype(str).unique()))

    def _ids_presentes_em_unicos(self, ids: set[str], df_unicos: pd.DataFrame) -> set[str]:
        """
        Restringe o conjunto 'ids' (vindos do arquivo de PARÂMETROS) aos IDs que
        realmente possuem cronograma no DF 'df_unicos' (RESULTADOS).
        O casamento é feito por ID normalizado para tolerar pequenas variações.
        """
        if not ids or df_unicos is None or df_unicos.empty:
            return set()
        if "id_norm" not in df_unicos.columns:
            # Segurança: se por algum motivo o DF de unicos não tem id_norm, não filtramos.
            return ids
        unicos_norm = set(map(str, df_unicos["id_norm"].astype(str).unique()))
        return {i for i in ids if IdUtil.normalizar_id(i) in unicos_norm}

    def _salvar_cronogramas_para_lado(self,
                                    param: str,
                                    ids: set[str],
                                    df_unicos: pd.DataFrame,
                                    nome_base_lado: str) -> None:
        """
        Gera '<param>_<base>.txt' com UMA LINHA por cronograma, copiando a linha
        bruta do '*_resultados_unicos.txt' do MESMO LADO (A ou B).
        A lista de 'ids' vem EXCLUSIVAMENTE do arquivo de PARÂMETROS correspondente.
        """
        if not ids or df_unicos is None or df_unicos.empty:
            return

        # Dicionários de busca no resultados_unicos:
        # - por ID original (como veio no arquivo de unicos)
        # - por ID canonizado (para casar com 'parâmetros' em casos como 66 vs 66.0)
        mapa_raw_por_id = {}
        mapa_raw_por_id_canon = {}
        for _, row in df_unicos[["id", "raw_line"]].iterrows():
            id_un = str(row["id"]).strip()
            raw = str(row["raw_line"])
            mapa_raw_por_id[id_un] = raw
            mapa_raw_por_id_canon[IdUtil.canon_param_id(id_un)] = raw

        # Ordenação estável: IDs numéricos primeiro (ordem crescente), depois alfanuméricos
        def _ord_key(v: str):
            v_can = IdUtil.canon_param_id(v)
            return (0, int(v_can)) if v_can.isdigit() else (1, v_can)

        ids_ordenados = sorted(ids, key=_ord_key)

        # Monta o arquivo copiando a linha bruta do resultados_unicos:
        linhas = []
        faltantes = []
        for pid in ids_ordenados:
            pid_can = IdUtil.canon_param_id(pid)
            if pid in mapa_raw_por_id:
                linhas.append(mapa_raw_por_id[pid])
            elif pid_can in mapa_raw_por_id_canon:
                linhas.append(mapa_raw_por_id_canon[pid_can])
            else:
                # Não escreve placeholder; apenas registra para aviso no console.
                faltantes.append(str(pid))

        safe_param = re.sub(r"[^\w\-]+", "_", param.strip())
        arq_out = self.dir_out / f"{safe_param}_{nome_base_lado}.txt"
        FileIO.escrever_texto(arq_out, "\n".join(linhas))

        # Aviso opcional em console (não interfere nos arquivos):
        if faltantes:
            print(f"[AVISO] {arq_out.name}: {len(faltantes)} ID(s) de parâmetros sem linha correspondente no resultados_unicos: {', '.join(map(str, faltantes))}")

    def _gerar_relatorio_texto(self) -> str:
        """
        Relatório geral por parâmetro, com contagens baseadas EXCLUSIVAMENTE
        nos arquivos de PARÂMETROS de cada lado (A e B).
        """
        a_name, b_name = self.arq_a_param.name, self.arq_b_param.name
        linhas, resumo = [], []
        linhas.append("COMPARATIVO DE PARÂMETROS ENTRE ARQUIVOS")
        linhas.append(f"Arquivo A: {a_name}")
        linhas.append(f"Arquivo B: {b_name}")
        linhas.append("Chave: coluna 'id'")
        linhas.append("")

        for p in self.params:
            set_a = self._valores_unicos_normalizados(self.df_a, p)
            set_b = self._valores_unicos_normalizados(self.df_b, p)
            diff_a = set_a - set_b
            diff_b = set_b - set_a

            ids_a = self._ids_por_valores(self.df_a, p, diff_a)  # IDs do lado A (parâmetros)
            ids_b = self._ids_por_valores(self.df_b, p, diff_b)  # IDs do lado B (parâmetros)

            if diff_a or diff_b:
                resumo.append(
                    f"- {p}: A_only valores={len(diff_a)} (IDs={len(ids_a)}) | "
                    f"B_only valores={len(diff_b)} (IDs={len(ids_b)})"
                )

        if resumo:
            linhas.append("Resumo (valores exclusivos e IDs afetados por parâmetro):")
            linhas.extend(resumo)
        else:
            linhas.append("Sem diferenças nos parâmetros listados.")
        linhas.append("")
        linhas.append("=" * 80)
        linhas.append("")

        for p in self.params:
            set_a = self._valores_unicos_normalizados(self.df_a, p)
            set_b = self._valores_unicos_normalizados(self.df_b, p)
            diff_a = set_a - set_b
            diff_b = set_b - set_a

            ids_a = self._ids_por_valores(self.df_a, p, diff_a)
            ids_b = self._ids_por_valores(self.df_b, p, diff_b)
            ord_a = ValueUtil.ordenar_safely(diff_a)
            ord_b = ValueUtil.ordenar_safely(diff_b)

            linhas.append(f"Parâmetro: {p}")
            linhas.append(f"  Valores apenas no Arquivo A ({a_name}) [n={len(ord_a)} | IDs={len(ids_a)}]:")
            linhas.append("    " + (", ".join(ord_a) if ord_a else "(nenhum)"))
            linhas.append(f"  Valores apenas no Arquivo B ({b_name}) [n={len(ord_b)} | IDs={len(ids_b)}]:")
            linhas.append("    " + (", ".join(ord_b) if ord_b else "(nenhum)"))
            linhas.append("")

        return "\n".join(linhas)

    def executar(self) -> Path:
        """
        1) Salva o relatório geral com as contagens vindas dos ARQUIVOS DE PARÂMETROS.
        2) Para cada parâmetro e para cada lado:
        - Obtém os IDs exclusivos daquele lado (com base apenas no 'parâmetros');
        - Copia as LINHAS BRUTAS correspondentes do '*_resultados_unicos.txt' do MESMO lado
            para '<param>_<base>.txt' (ex.: 'folga_R28_tarefas_4.txt').
        """
        # Relatório
        rel_path = self.dir_out / f"{self.base_rel}.txt"
        FileIO.escrever_texto(rel_path, self._gerar_relatorio_texto())

        # Geração dos arquivos por parâmetro e lado
        for p in self.params:
            set_a = self._valores_unicos_normalizados(self.df_a, p)
            set_b = self._valores_unicos_normalizados(self.df_b, p)
            diff_a = set_a - set_b
            diff_b = set_b - set_a

            # IDs exclusivos por LADO, obtidos APENAS dos arquivos de PARÂMETROS
            if diff_a:
                ids_a = self._ids_por_valores(self.df_a, p, diff_a)   # lado A
                self._salvar_cronogramas_para_lado(p, ids_a, self.df_a_un, self.base_a)

            if diff_b:
                ids_b = self._ids_por_valores(self.df_b, p, diff_b)   # lado B
                self._salvar_cronogramas_para_lado(p, ids_b, self.df_b_un, self.base_b)

        return self.dir_out

class App:
    """Ponto de entrada do script."""

    @staticmethod
    def run():
        cmp_ = ComparadorParametros(
            Config.ARQ_A_PARAM, Config.ARQ_B_PARAM,
            Config.ARQ_A_UNICOS, Config.ARQ_B_UNICOS,
            Config.PARAMS, id_col="id"
        )
        out = cmp_.executar()
        print(f"Pasta de saída: {out}")
        print(f"Relatório geral: {out / (cmp_.base_rel + '.txt')}")

if __name__ == "__main__":
    App.run()