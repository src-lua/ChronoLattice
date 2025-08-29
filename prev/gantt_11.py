import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib import colormaps
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path
import ast
import os
from collections import defaultdict

# --- CONSTANTES ---
# Define constantes para facilitar a manutenção de valores fixos.
BAR_HEIGHT = 0.6
MANDATORY_DEP_COLOR = 'black'
ARBITRARY_DEP_COLOR = 'gray'
BASE_SCHEDULE_ID = '0 (Base)'


class GanttPlotter:
    """
    Responsável por toda a plotagem de gráficos (Gantt e Histograma de Recursos)
    usando Matplotlib. As cores das tarefas e do histograma são unificadas por recurso.
    """
    def __init__(self, figure):
        self.figure = figure
        self.DEFAULT_TASK_COLOR = 'lightgrey'

    def clear(self):
        """Limpa a figura para desenhar um novo gráfico."""
        self.figure.clear()

    def draw(self):
        """Redesenha o canvas que contém a figura."""
        self.figure.canvas.draw()

    def _determine_resource_order(self, gantt_data, base_tasks_map):
        """
        Determina a ordem de empilhamento dos recursos no histograma com base
        no primeiro momento em que cada recurso é utilizado no cronograma.
        """
        first_usage = {}
        # Itera sobre as tarefas ordenadas por tempo de início
        for task in sorted(gantt_data, key=lambda x: x['Start']):
            task_id = task['Task']
            base_info = base_tasks_map.get(task_id)
            if base_info and len(base_info) > 4:
                resource_id = base_info[4]
                if resource_id not in first_usage:
                    first_usage[resource_id] = task['Start']

        # Retorna a lista de recursos ordenada pelo tempo de primeiro uso (e nome como desempate)
        return sorted(first_usage.keys(), key=lambda res: (first_usage[res], res))

    def plot_charts(self, schedule_id, gantt_data, raw_tasks, base_tasks_map, sort_mode='id'):
        """
        Coordena a plotagem do Gráfico de Gantt e do Histograma de Recursos.
        """
        if not gantt_data:
            self.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, "Nenhum dado de cronograma para exibir.", ha='center', va='center')
            self.draw()
            return

        self.clear()
        
        all_possible_resources = sorted(list(set(task[4] for task in base_tasks_map.values() if len(task) > 4)))
        cmap = colormaps['tab10']
        resource_color_map = {res: cmap(i % cmap.N) for i, res in enumerate(all_possible_resources)}

        resources_in_usage_order = self._determine_resource_order(gantt_data, base_tasks_map)

        gs = self.figure.add_gridspec(2, 1, height_ratios=[2, 1])
        ax_gantt = self.figure.add_subplot(gs[0, 0])
        ax_hist = self.figure.add_subplot(gs[1, 0], sharex=ax_gantt)
        
        plt.setp(ax_gantt.get_xticklabels(), visible=False)

        self._plot_gantt_chart(ax_gantt, schedule_id, gantt_data, raw_tasks, base_tasks_map, resource_color_map, sort_mode)
        self._plot_resource_histogram(ax_hist, gantt_data, base_tasks_map, resources_in_usage_order, resource_color_map)
        
        self.figure.tight_layout()
        self.draw()

    def _sort_gantt_data(self, gantt_data, sort_mode):
        """Ordena os dados do Gantt com base no modo selecionado."""
        if sort_mode == 'layer':
            gantt_data.sort(key=lambda item: (item.get('Layer', 0), item['Duration'], item['Task']))
        elif sort_mode == 'network':
            gantt_data.sort(key=lambda item: (item.get('NetworkDuration', 0), item.get('NetworkID', ''), item['Start'], item['Task']))
        else:  # 'id'
            try:
                gantt_data.sort(key=lambda item: int(item['Task'][1:]))
            except (ValueError, IndexError):
                gantt_data.sort(key=lambda item: item['Task'])
        return gantt_data

    def _draw_gantt_bars_and_labels(self, ax, gantt_data, y_pos_map, task_colors):
        """Desenha as barras do gráfico de Gantt e as etiquetas de duração."""
        tasks_labels = [item['Task'] for item in gantt_data]
        ax.barh(
            tasks_labels, 
            [item['Duration'] for item in gantt_data], 
            left=[item['Start'] for item in gantt_data], 
            color=task_colors,
            edgecolor='black', 
            height=BAR_HEIGHT
        )
        for task in gantt_data:
            y_pos = y_pos_map[task['Task']]
            ax.text(
                task['Start'] + task['Duration'] / 2, y_pos, str(task['Duration']), 
                ha='center', va='center', color='white', fontweight='bold'
            )

    def _draw_all_dependency_arrows(self, ax, raw_tasks, task_data_map, y_pos_map):
        """Itera sobre as tarefas para desenhar todas as setas de dependência."""
        for successor_task in raw_tasks:
            if len(successor_task) < 4: continue
            succ_id, _, mandatory_preds_str, arbitrary_preds_str = successor_task

            mandatory_preds = [p.strip() for p in mandatory_preds_str.split(',') if p.strip()]
            for pred_id in mandatory_preds:
                self._draw_dependency_arrow(ax, succ_id, pred_id, task_data_map, y_pos_map, is_arbitrary=False)

            arbitrary_preds = [p.strip() for p in arbitrary_preds_str.split(',') if p.strip()]
            for pred_id in arbitrary_preds:
                self._draw_dependency_arrow(ax, succ_id, pred_id, task_data_map, y_pos_map, is_arbitrary=True)

    def _plot_gantt_chart(self, ax, schedule_id, gantt_data, raw_tasks, base_tasks_map, resource_color_map, sort_mode):
        """Cria o Gráfico de Gantt orquestrando as funções auxiliares."""
        sorted_data = self._sort_gantt_data(gantt_data, sort_mode)
        y_pos_map = {item['Task']: i for i, item in enumerate(sorted_data)}
        task_data_map = {item['Task']: item for item in sorted_data}
        
        task_colors = []
        for item in sorted_data:
            base_task_info = base_tasks_map.get(item['Task'])
            if base_task_info and len(base_task_info) > 4:
                resource_id = base_task_info[4]
                task_colors.append(resource_color_map.get(resource_id, self.DEFAULT_TASK_COLOR))
            else:
                task_colors.append(self.DEFAULT_TASK_COLOR)

        self._draw_gantt_bars_and_labels(ax, sorted_data, y_pos_map, task_colors)
        self._draw_all_dependency_arrows(ax, raw_tasks, task_data_map, y_pos_map)

        ax.invert_yaxis()
        ax.set_ylabel('Tarefas')
        ax.set_title(f'Gráfico de Gantt para o Cronograma: {schedule_id}')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        legend_elements = [
            Line2D([0], [0], color=MANDATORY_DEP_COLOR, lw=1.5, label='Dependência Obrigatória'),
            Line2D([0], [0], color=ARBITRARY_DEP_COLOR, lw=1.5, linestyle='--', label='Dependência Arbitrária')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def _draw_dependency_arrow(self, ax, succ_id, pred_id, task_data_map, y_pos_map, is_arbitrary):
        """Desenha uma única seta de dependência entre duas tarefas."""
        if not pred_id or pred_id not in task_data_map or succ_id not in task_data_map:
            return

        color = ARBITRARY_DEP_COLOR if is_arbitrary else MANDATORY_DEP_COLOR
        linestyle = '--' if is_arbitrary else '-'
        pred_data, succ_data = task_data_map[pred_id], task_data_map[succ_id]
        
        px_out = pred_data['Finish']
        py_out = y_pos_map[pred_id] + (BAR_HEIGHT / 6 if is_arbitrary else -BAR_HEIGHT / 6)
        sx_in = succ_data['Start']
        sy_in = y_pos_map[succ_id] + (BAR_HEIGHT / 2 if is_arbitrary else -BAR_HEIGHT / 2)

        path_data = [(px_out, py_out), (px_out + 0.2, py_out), (px_out + 0.2, sy_in), (sx_in, sy_in)]
        
        arrow = FancyArrowPatch(
            path=Path(path_data), arrowstyle='-|>,head_length=5,head_width=3',
            color=color, linestyle=linestyle, lw=1.0, shrinkA=0, shrinkB=2
        )
        ax.add_patch(arrow)

    def _plot_resource_histogram(self, ax, gantt_data, base_tasks_map, resources_to_plot, resource_color_map):
        """Cria o Histograma de Uso de Recursos."""
        if not base_tasks_map or not resources_to_plot: return

        makespan = max((task['Finish'] for task in gantt_data), default=0)
        if makespan == 0: return

        resource_usage = {res: [0] * makespan for res in resources_to_plot}
        
        for task in gantt_data:
            base_task_info = base_tasks_map.get(task['Task'])
            if base_task_info and len(base_task_info) >= 6:
                resource_id, resource_qty = base_task_info[4], base_task_info[5]
                if resource_id in resource_usage:
                    for day in range(task['Start'], task['Finish']):
                        if day < makespan:
                            resource_usage[resource_id][day] += resource_qty
        
        time_axis = range(makespan)
        bottom = [0] * makespan
        
        for resource_id in resources_to_plot:
            color = resource_color_map.get(resource_id)
            if color:
                ax.bar(time_axis, resource_usage[resource_id], bottom=bottom, color=color, label=resource_id, align='center')
                bottom = [b + u for b, u in zip(bottom, resource_usage[resource_id])]

        ax.set_ylabel('Qtd. Recursos')
        ax.set_title('Histograma de Uso de Recursos')
        ax.legend(title='Recursos')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        ax.set_xticks(time_axis)
        ax.set_xticklabels([str(t + 1) for t in time_axis])
        ax.set_xlabel('Tempo')

class ScheduleDataManager:
    """
    Gerencia o carregamento, análise (parsing) e cálculo dos dados dos cronogramas.
    """
    def __init__(self):
        self.base_tasks_map = {}
        self.all_schedules = {}
        self.durations_map = defaultdict(list)
        self.task_layers_map = {}
        self.task_network_map = {}
        self.network_durations = {}

    def _calculate_layers_from_base(self):
        """Calcula a camada de cada tarefa com base nas predecessoras OBRIGATÓRIAS."""
        if not self.base_tasks_map:
            self.task_layers_map = {}
            return

        tasks = list(self.base_tasks_map.values())
        task_map = {task[0]: task for task in tasks}
        dependencies = {task[0]: [] for task in tasks}
        in_degree = {task[0]: 0 for task in tasks}

        for task_id, _, mandatory_preds_str, *_ in tasks:
            mandatory_preds = [p.strip() for p in mandatory_preds_str.split(',') if p.strip()]
            for p_id in mandatory_preds:
                if p_id in task_map:
                    dependencies[p_id].append(task_id)
                    in_degree[task_id] += 1
        
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        layers = {task_id: 0 for task_id in queue}
        
        head = 0
        while head < len(queue):
            current_task_id = queue[head]; head += 1
            current_layer = layers.get(current_task_id, 0)
            for dependent_task_id in dependencies[current_task_id]:
                layers[dependent_task_id] = max(layers.get(dependent_task_id, 0), current_layer + 1)
                in_degree[dependent_task_id] -= 1
                if in_degree[dependent_task_id] == 0:
                    queue.append(dependent_task_id)
        self.task_layers_map = layers

    def _calculate_networks_from_base(self):
        """Identifica redes de tarefas e calcula a duração total de cada uma."""
        if not self.base_tasks_map: return

        tasks = list(self.base_tasks_map.values())
        task_map = {task[0]: task for task in tasks}
        successors = {task[0]: [] for task in tasks}
        in_degree = {task[0]: 0 for task in tasks}

        for task_id, _, mandatory_preds_str, *_ in tasks:
            mandatory_preds = [p.strip() for p in mandatory_preds_str.split(',') if p.strip()]
            for p_id in mandatory_preds:
                if p_id in task_map:
                    successors[p_id].append(task_id)
                    in_degree[task_id] += 1
        
        source_nodes = [task_id for task_id, degree in in_degree.items() if degree == 0]
        network_map, network_durations = {}, {}

        for source_node_id in source_nodes:
            network_id, network_total_duration = f"Rede_{source_node_id}", 0
            q, visited = [source_node_id], {source_node_id}
            
            while q:
                current_task_id = q.pop(0)
                network_map[current_task_id] = network_id
                network_total_duration += task_map[current_task_id][1]
                for succ_id in successors.get(current_task_id, []):
                    if succ_id not in visited:
                        visited.add(succ_id); q.append(succ_id)
            network_durations[network_id] = network_total_duration

        self.task_network_map = network_map
        self.network_durations = network_durations

    def load_base_schedule(self, file_content):
        """Carrega e analisa o arquivo de cronograma base."""
        try:
            data = ast.literal_eval(file_content)
            if not isinstance(data, list):
                raise TypeError("O conteúdo do arquivo base não é uma lista válida.")
            
            self.base_tasks_map = {task[0]: task for task in data}
            base_schedule_tasks = [(t[0], t[1], t[2], t[3]) for t in data]
            self.all_schedules[BASE_SCHEDULE_ID] = base_schedule_tasks
            
            self._calculate_layers_from_base()
            self._calculate_networks_from_base()
            return True
        except Exception as e:
            messagebox.showerror("Erro de Leitura (Arquivo Base)", f"Falha ao processar o arquivo base: {e}")
            return False

    def load_schedules_list(self, file_content):
        """Carrega e analisa o arquivo com a lista de múltiplos cronogramas."""
        parsed_schedules = {}
        for i, line in enumerate(file_content.strip().split('\n')):
            if not line.strip(): continue
            try:
                item = ast.literal_eval(line.strip())
                if isinstance(item, list) and len(item) > 1:
                    parsed_schedules[str(item[0])] = item[1:]
                else:
                    messagebox.showwarning("Formato Inesperado", f"A linha {i+1} tem um formato inválido.")
            except (ValueError, SyntaxError) as e:
                messagebox.showerror("Erro de Leitura", f"Erro de sintaxe na linha {i+1}: {e}")
                return False
        self.all_schedules = {k: v for k, v in self.all_schedules.items() if k == BASE_SCHEDULE_ID}
        self.all_schedules.update(parsed_schedules)
        return True

    def calculate_all_makespans(self):
        """Calcula e agrupa todos os cronogramas por sua duração (makespan)."""
        self.durations_map.clear()
        for schedule_id, tasks in self.all_schedules.items():
            try:
                gantt_data = self.calculate_gantt_data(tasks)
                if gantt_data:
                    makespan = max(task['Finish'] for task in gantt_data)
                    self.durations_map[makespan].append(schedule_id)
            except ValueError as e:
                print(f"Aviso: {e} no cronograma ID {schedule_id}. Ignorando.")

    def _initialize_dependency_graph(self, tasks):
        """Prepara as estruturas de dados para o cálculo do Gantt."""
        task_map = {task[0]: task for task in tasks}
        dependencies = {task[0]: [] for task in tasks}
        in_degree = {task[0]: 0 for task in tasks}

        for task_id, _, pred1_str, pred2_str in tasks:
            all_preds = [p.strip() for p in pred1_str.split(',') if p.strip()] + \
                        [p.strip() for p in pred2_str.split(',') if p.strip()]
            for p_id in all_preds:
                if p_id in task_map:
                    dependencies[p_id].append(task_id)
                    in_degree[task_id] += 1
        return task_map, dependencies, in_degree

    def calculate_gantt_data(self, tasks):
        """Calcula os tempos de início e fim para cada tarefa."""
        if not tasks: return []
        
        task_map, dependencies, in_degree = self._initialize_dependency_graph(tasks)
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        start_times, finish_times = {}, {}
        gantt_chart_data = []
        
        while queue:
            current_task_id = queue.pop(0)
            task_info = task_map[current_task_id]
            duration, pred1_str, pred2_str = task_info[1], task_info[2], task_info[3]
            
            all_preds = [p.strip() for p in pred1_str.split(',') if p.strip()] + \
                        [p.strip() for p in pred2_str.split(',') if p.strip()]
            
            finished_preds = [p_id for p_id in all_preds if p_id in finish_times]
            start_time = max([finish_times.get(p_id, 0) for p_id in finished_preds] if finished_preds else [0])
            finish_time = start_time + duration
            
            start_times[current_task_id] = start_time
            finish_times[current_task_id] = finish_time
            
            gantt_chart_data.append({'Task': current_task_id, 'Start': start_time, 'Duration': duration, 'Finish': finish_time})

            for dependent_task_id in dependencies.get(current_task_id, []):
                in_degree[dependent_task_id] -= 1
                if in_degree[dependent_task_id] == 0:
                    queue.append(dependent_task_id)

        if len(gantt_chart_data) < len(tasks):
            raise ValueError("Ciclo detectado nas dependências")
        return gantt_chart_data

    def get_annotated_gantt_data(self, schedule_id):
        """Calcula e anota os dados do Gantt com informações de camada e rede."""
        tasks = self.all_schedules[schedule_id]
        gantt_data = self.calculate_gantt_data(tasks)

        for task_dict in gantt_data:
            task_id = task_dict['Task']
            task_dict['Layer'] = self.task_layers_map.get(task_id, 0)
            network_id = self.task_network_map.get(task_id, "N/A")
            task_dict['NetworkID'] = network_id
            task_dict['NetworkDuration'] = self.network_durations.get(network_id, float('inf'))
            
        return gantt_data, tasks

class GanttApp:
    """
    Classe principal da aplicação. Gerencia a interface gráfica (Tkinter) e
    a interação com o usuário.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Cronogramas e Recursos")
        self.root.state('zoomed')

        self.data_manager = ScheduleDataManager()
        self.plotter = GanttPlotter(Figure(figsize=(10, 8), dpi=100))
        
        self.parametros_data = []
        self.param_header = []
        self.resource_columns = []

        # Widgets para o novo filtro de recursos
        self.resource_filter_combo = None
        self.resource_filter_entry = None

        self._is_updating_filters = False
        self.sort_mode = 'id'

        self._setup_ui()
        self._connect_events()

    # --- Métodos de Construção da UI ---

    def _setup_ui(self):
        """Cria e organiza a interface com painéis de controle de largura fixa."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_controls_panel = self._create_left_control_panel(main_frame)
        left_controls_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        right_controls_panel = self._create_right_control_panel(main_frame)
        right_controls_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))

        chart_frame = ttk.Frame(main_frame, padding="10")
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._populate_chart_panel(chart_frame)

    def _populate_chart_panel(self, parent_frame):
        """Popula o frame do gráfico com o canvas e a barra de ferramentas."""
        canvas = FigureCanvasTkAgg(self.plotter.figure, master=parent_frame)
        toolbar = NavigationToolbar2Tk(canvas, parent_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _create_left_control_panel(self, parent):
        """Cria o painel esquerdo com carregamento, filtros e recursos."""
        panel = ttk.Frame(parent, width=320)
        panel.pack_propagate(False)

        self._create_load_frame(panel)
        self._create_filter_options_frame(panel)
        self._create_resources_frame(panel)
        
        return panel

    def _create_right_control_panel(self, parent):
        """Cria o painel direito (central) com a lista de cronogramas e ações."""
        panel = ttk.Frame(parent)
        self._create_schedule_list_frame(panel)
        return panel

    def _create_load_frame(self, parent):
        """Cria o frame para carregamento de todos os arquivos."""
        frame = ttk.LabelFrame(parent, text="Carregamento de Arquivos", padding=10)
        frame.pack(fill=tk.X, anchor='n', pady=(0,5))
        frame.columnconfigure(1, weight=1)

        self.load_base_btn = ttk.Button(frame, text="Cronograma Base...")
        self.load_base_btn.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.base_file_label = ttk.Label(frame, text="Nenhum arquivo", foreground="gray", wraplength=180)
        self.base_file_label.grid(row=0, column=1, padx=5, sticky="w")
        
        self.load_schedules_btn = ttk.Button(frame, text="Lista de Cronogramas...")
        self.load_schedules_btn.grid(row=1, column=0, sticky="ew", pady=2)
        self.schedules_file_label = ttk.Label(frame, text="Nenhum arquivo", foreground="gray", wraplength=180)
        self.schedules_file_label.grid(row=1, column=1, padx=5, sticky="w")

        self.load_params_btn = ttk.Button(frame, text="Parâmetros de Cronograma...", command=self.load_parameters)
        self.load_params_btn.grid(row=2, column=0, sticky="ew", pady=(2, 0))
        self.params_file_label = ttk.Label(frame, text="Nenhum arquivo", foreground="gray", wraplength=180)
        self.params_file_label.grid(row=2, column=1, padx=5, sticky="w")

    def _create_filter_options_frame(self, parent):
        """Cria o frame com as opções de filtro e ordenação."""
        frame = ttk.LabelFrame(parent, text="Filtros", padding=10)
        frame.pack(fill=tk.X, anchor='n')

        ttk.Label(frame, text="Filtrar por Duração:").pack(anchor="w")
        self.duration_combobox = ttk.Combobox(frame, state="readonly", width=25)
        self.duration_combobox.pack(fill=tk.X, pady=(0, 5))
        
        param_keys = ["folga", "linearidade", "expansão", "regularidade"]
        self.param_comboboxes = {}
        for key in param_keys:
            ttk.Label(frame, text=f"Filtro Parâmetro {key}:").pack(anchor="w")
            combo = ttk.Combobox(frame, state="disabled")
            combo.pack(fill=tk.X, pady=(0, 5))
            self.param_comboboxes[key] = combo

        self.sort_button = ttk.Button(frame, text="Ordenar por ID", command=self._toggle_sort_mode)
        self.sort_button.pack(fill=tk.X, pady=(10, 0))

    def _create_resources_frame(self, parent):
        """Cria o frame para exibir e filtrar por recursos."""
        frame = ttk.LabelFrame(parent, text="Recursos", padding=10)
        frame.pack(fill=tk.X, anchor='n', pady=(5,0))

        # --- 1. Seção de Exibição (mostra os valores máximos atuais) ---
        ttk.Label(frame, text="Valores Máximos (nos cenários filtrados):").pack(anchor="w")
        
        list_container = ttk.Frame(frame)
        list_container.pack(fill=tk.X, expand=True, pady=(2, 10))
        self.resources_listbox = tk.Listbox(list_container, height=4)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.resources_listbox.yview)
        self.resources_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.resources_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- 2. Seção de Filtragem (permite definir um limite) ---
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=5)
        ttk.Label(frame, text="Filtrar por Limite de Recurso:").pack(anchor="w")

        self.resource_filter_combo = ttk.Combobox(frame, state="readonly")
        self.resource_filter_combo.pack(fill=tk.X, pady=(2,5))
        
        self.resource_filter_entry = ttk.Entry(frame)
        self.resource_filter_entry.pack(fill=tk.X, pady=5)
        
        apply_btn = ttk.Button(frame, text="Aplicar Filtro de Recursos", command=self._update_cascading_filters)
        apply_btn.pack(fill=tk.X, pady=5)

        return frame

    def _create_schedule_list_frame(self, parent):
        """Cria o frame com a lista de cronogramas, contador e botão salvar."""
        frame = ttk.LabelFrame(parent, text="Seleção de Cronogramas", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        self.save_btn = ttk.Button(frame, text="Salvar Cronogramas Filtrados...", command=self._save_filtered_schedules)
        self.save_btn.pack(fill=tk.X, pady=(5,0), side=tk.BOTTOM)
        
        self.count_label = ttk.Label(frame, text="Total de Cronogramas: 0")
        self.count_label.pack(side=tk.BOTTOM, pady=(5,0))

        ttk.Label(frame, text="Selecione um ID de Cronograma:").pack(anchor="w")
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(2,0))
        
        self.schedule_listbox = tk.Listbox(list_frame, selectmode=tk.BROWSE, exportselection=False, width=30)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.schedule_listbox.yview)
        self.schedule_listbox.config(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.schedule_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # --- Métodos de Eventos e Lógica da UI ---

    def _connect_events(self):
        """Conecta os eventos dos widgets às suas funções de callback."""
        self.load_base_btn.config(command=self._load_base_file)
        self.load_schedules_btn.config(command=self._load_schedules_file)
        self.save_btn.config(command=self._save_filtered_schedules)
        self.schedule_listbox.bind("<<ListboxSelect>>", self._on_schedule_select)
        
        self.duration_combobox.bind("<<ComboboxSelected>>", self._update_cascading_filters)
        for combo in self.param_comboboxes.values():
            combo.bind("<<ComboboxSelected>>", self._update_cascading_filters)

    def _load_base_file(self):
        """Carrega o arquivo de cronograma base e atualiza a UI."""
        file_path = filedialog.askopenfilename(title="Selecione o ARQUIVO BASE", filetypes=[("Arquivos de Texto", "*.txt")])
        if not file_path: return
        
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        
        if hasattr(self, 'resources_listbox'):
            self.resources_listbox.delete(0, tk.END)

        if self.data_manager.load_base_schedule(content):
            self.base_file_label.config(text=os.path.basename(file_path), foreground="green")
            self._update_ui_after_load()
            self._update_resources_list()

    def _load_schedules_file(self):
        file_path = filedialog.askopenfilename(title="Selecione a LISTA de cronogramas", filetypes=[("Arquivos de Texto", "*.txt")])
        if not file_path: return
        with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        if self.data_manager.load_schedules_list(content):
            self.schedules_file_label.config(text=os.path.basename(file_path), foreground="green")
            self.root.title(f"Visualizador - {os.path.basename(file_path)}")
            self._update_ui_after_load()

    def load_parameters(self):
        """Carrega e processa o arquivo de parâmetros como um CSV com cabeçalho."""
        import csv
        filepath = filedialog.askopenfilename(title="Selecione o arquivo de parâmetros", filetypes=(("Arquivos de Texto", "*.txt"),))
        if not filepath: return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.param_header = next(reader)
                self.resource_columns = [h for h in self.param_header if h.endswith('_max')]
                self.parametros_data = []
                for row_values in reader:
                    row_dict = {}
                    for i, header in enumerate(self.param_header):
                        try:
                            row_dict[header] = float(row_values[i])
                        except (ValueError, IndexError):
                            row_dict[header] = row_values[i] if i < len(row_values) else None
                    self.parametros_data.append(row_dict)
            self.params_file_label.config(text=os.path.basename(filepath), foreground="green")
            self._update_cascading_filters()
        except Exception as e:
            messagebox.showerror("Erro ao Processar Parâmetros", f"Ocorreu um erro: {e}")
            self.params_file_label.config(text="Falha ao carregar", foreground="red")

    def _update_ui_after_load(self):
        """Atualiza a UI após um carregamento de dados."""
        if not self.data_manager.all_schedules: return
        self.data_manager.calculate_all_makespans()
        sorted_durations = sorted(self.data_manager.durations_map.keys())
        self.duration_combobox['values'] = ["Mostrar Todos"] + sorted_durations
        self.duration_combobox.current(0)
        self._on_duration_selected()

    def _on_duration_selected(self, event=None):
        """Filtra a lista de IDs com base na duração selecionada."""
        self.schedule_listbox.delete(0, tk.END)
        selected_duration = self.duration_combobox.get()
        ids_to_show = []
        if selected_duration == "Mostrar Todos":
            ids_to_show = sorted(self.data_manager.all_schedules.keys(), key=lambda k: int(k) if k.isdigit() else float('inf'))
            if BASE_SCHEDULE_ID in ids_to_show:
                ids_to_show.remove(BASE_SCHEDULE_ID); ids_to_show.insert(0, BASE_SCHEDULE_ID)
        else:
            ids_to_show = sorted(self.data_manager.durations_map.get(int(selected_duration), []))
        
        self._populate_schedule_listbox(ids_to_show)

    def _populate_schedule_listbox(self, ids_to_display):
        """Limpa e preenche a listbox de cronogramas com os IDs fornecidos."""
        self.schedule_listbox.delete(0, tk.END)
        for schedule_id in sorted(ids_to_display, key=lambda sid: (0, int(sid)) if sid.isdigit() else (1, sid)):
            self.schedule_listbox.insert(tk.END, schedule_id)
        self._update_schedule_count()
        if self.schedule_listbox.size() > 0:
            self.schedule_listbox.select_set(0)
            self.schedule_listbox.event_generate("<<ListboxSelect>>")
        else:
            self.plotter.clear(); self.plotter.draw()

    def _on_schedule_select(self, event=None):
        """Atualiza os gráficos ao selecionar um cronograma na lista."""
        if not self.schedule_listbox.curselection(): return
        selected_id = self.schedule_listbox.get(self.schedule_listbox.curselection()[0])
        try:
            gantt_data, raw_tasks = self.data_manager.get_annotated_gantt_data(selected_id)
            self.plotter.plot_charts(selected_id, gantt_data, raw_tasks, self.data_manager.base_tasks_map, self.sort_mode)
        except Exception as e:
            messagebox.showerror("Erro ao Gerar Gráfico", f"Não foi possível gerar o gráfico para o ID {selected_id}:\n{e}")
            self.plotter.clear(); self.plotter.draw()

    def _save_filtered_schedules(self):
        """Salva os cronogramas atualmente visíveis na listbox."""
        current_ids = self.schedule_listbox.get(0, tk.END)
        if not current_ids:
            messagebox.showwarning("Nada a Salvar", "A lista de cronogramas filtrados está vazia.")
            return
        file_path = filedialog.asksaveasfilename(title="Salvar Cronogramas Filtrados", defaultextension=".txt", filetypes=[("Arquivos de Texto", "*.txt")])
        if not file_path: return
        try:
            output = []
            for sid in current_ids:
                tasks = self.data_manager.all_schedules[sid]
                formatted_tasks = ', '.join(map(repr, tasks))
                output.append(f"[{repr(sid)}, {formatted_tasks}]")
            with open(file_path, 'w', encoding='utf-8') as f: f.write('\n'.join(output))
            messagebox.showinfo("Sucesso", f"Cronogramas salvos em: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Erro ao Salvar", f"Ocorreu um erro: {e}")
    
    def _update_cascading_filters(self, event=None):
        """
        Atualiza todos os filtros (parâmetros e recursos) e a UI de acordo.
        """
        if self._is_updating_filters: return
        self._is_updating_filters = True

        def get_possible_rows(active_filters):
            # Coleta os valores de todos os filtros de parâmetro
            selections = {
                'duração': self.duration_combobox.get(),
                'folga': self.param_comboboxes['folga'].get(),
                'linearidade': self.param_comboboxes['linearidade'].get(),
                'expansão': self.param_comboboxes['expansão'].get(),
                'regularidade': self.param_comboboxes['regularidade'].get(),
            }
            
            # Coleta os valores do novo filtro de recurso
            res_key_to_filter = self.resource_filter_combo.get() if self.resource_filter_combo else ""
            res_limit_str = self.resource_filter_entry.get() if self.resource_filter_entry else ""
            res_limit_val = None 
            if res_limit_str:
                try:
                    res_limit_val = float(res_limit_str)
                except ValueError:
                    pass # Ignora entrada inválida (não numérica)

            possible_data = []
            for row_dict in self.parametros_data:
                valid = True
                # 1. Aplica filtros de parâmetros
                for key, sel_value in selections.items():
                    if active_filters.get(key) and sel_value not in ('', 'Todos') and str(row_dict.get(key)) != sel_value:
                        valid = False
                        break
                if not valid: continue

                # 2. Aplica o filtro de recurso, se estiver ativo
                if res_key_to_filter and res_limit_val is not None:
                    resource_val = float(row_dict.get(res_key_to_filter, 0))
                    if resource_val > res_limit_val:
                        valid = False
                
                if valid:
                    possible_data.append(row_dict)
            return possible_data

        def update_combobox_options(combobox, data_key):
            # (Esta função aninhada permanece inalterada)
            active_filters = {k: (k != data_key) for k in self.param_comboboxes.keys()}
            active_filters['duração'] = (data_key != 'duração')
            possible_rows = get_possible_rows(active_filters)
            current_selection = combobox.get()
            unique_values = sorted(list(set(row[data_key] for row in possible_rows if data_key in row)))
            combobox['values'] = ['Todos'] + [str(v) for v in unique_values]
            combobox.set(current_selection if current_selection in combobox['values'] else 'Todos')
            combobox.config(state="readonly")

        # --- Atualização da UI ---
        update_combobox_options(self.duration_combobox, 'duração')
        for key, combo in self.param_comboboxes.items():
            update_combobox_options(combo, key)

        final_active_filters = {key: True for key in self.param_comboboxes.keys()}
        final_active_filters['duração'] = True
        final_rows = get_possible_rows(final_active_filters)
        
        id_key = self.param_header[0] if self.param_header else 'id'
        final_ids = {str(row[id_key]) for row in final_rows}
        loaded_schedule_ids = set(self.data_manager.all_schedules.keys())
        visible_ids = list(final_ids.intersection(loaded_schedule_ids))
        
        self._populate_schedule_listbox(visible_ids)
        self._update_resources_list()
        self._is_updating_filters = False

    def _update_resources_list(self):
        """
        Calcula e exibe a capacidade máxima de cada recurso e popula o combobox de filtro.
        """
        self.resources_listbox.delete(0, tk.END)
        
        # Popula o combobox para o filtro de recursos
        if self.resource_columns and self.resource_filter_combo:
             # Adiciona uma opção em branco para desativar o filtro
            self.resource_filter_combo['values'] = [""] + self.resource_columns

        if not self.parametros_data or not self.resource_columns: return

        visible_ids = set(self.schedule_listbox.get(0, tk.END))
        id_key = self.param_header[0] if self.param_header else 'id'
        filtered_param_data = [row for row in self.parametros_data if str(row.get(id_key)) in visible_ids]

        if not filtered_param_data:
            for resource in self.resource_columns:
                self.resources_listbox.insert(tk.END, f"{resource}: N/A")
            return

        for resource in self.resource_columns:
            values = [row.get(resource, 0) for row in filtered_param_data]
            max_value = max(values) if values else 0
            display_value = int(max_value) if max_value == int(max_value) else max_value
            self.resources_listbox.insert(tk.END, f"{resource}: {display_value}")

    def _toggle_sort_mode(self):
        """Alterna o modo de ordenação do gráfico de Gantt."""
        if self.sort_mode == 'id':
            self.sort_mode = 'layer'
            self.sort_button.config(text="Ordenar por Camada")
        elif self.sort_mode == 'layer':
            self.sort_mode = 'network'
            self.sort_button.config(text="Ordenar por Rede")
        else:
            self.sort_mode = 'id'
            self.sort_button.config(text="Ordenar por ID")
        self._on_schedule_select()

    def _update_schedule_count(self):
        """Atualiza o label com a contagem de cronogramas visíveis."""
        self.count_label.config(text=f"Total de Cronogramas: {self.schedule_listbox.size()}")
        """Atualiza o label com a contagem de cronogramas visíveis."""
        self.count_label.config(text=f"Total de Cronogramas: {self.schedule_listbox.size()}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GanttApp(root)
    root.mainloop()