import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image


class IlluminationCalculator:
    """Tkinter-приложение для расчета и визуализации освещенности на плоскости."""

    def __init__(self, root):
        """Инициализирует корневое окно и создаёт интерфейс с параметрами по умолчанию."""
        self.root = root
        self.root.title("Расчет освещенности на плоскости")
        self.root.geometry("1400x900")
        
        self.setup_default_parameters()
        self.setup_ui()
    
    # ------------------------------ ПАРАМЕТРЫ ПО УМОЛЧАНИЮ ----------------------------- #
        
    def setup_default_parameters(self):
        """Начальные значения области, источника и круга исследования."""
        self.W = 5000.0       # ширина области (мм)
        self.H = 5000.0       # высота области (мм)
        self.Wres = 400       # разрешение по X (пикс)
        self.Hres = 400       # разрешение по Y (пикс)
        self.xL = 0.0         # координаты ламбертовского источника
        self.yL = 0.0
        self.zL = 2000.0      # высота источника над плоскостью
        self.I0 = 1000.0      # сила излучения (Вт/ср)
        self.circle_x = 0.0   # центр исследуемого круга
        self.circle_y = 0.0
        self.circle_r = 2000.0
        
    # ------------------------------ UI ----------------------------- #
        
    def setup_ui(self):
        """Формирует главные фреймы, панель параметров, графики и поле с результатами."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=2)
        
        results_frame = ttk.LabelFrame(main_frame, text="Расчетные значения освещенности", padding="10")
        results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10), pady=(0, 10))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        self.results_text = tk.Text(results_frame, height=12, width=40, wrap=tk.WORD, font=("Courier", 9))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_text.config(state=tk.DISABLED)
        
        controls_frame = ttk.Frame(main_frame)
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.rowconfigure(0, weight=1)
        controls_frame.rowconfigure(1, weight=0)
        
        params_frame = ttk.LabelFrame(controls_frame, text="Параметры", padding="5")
        params_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        params_frame.columnconfigure(0, weight=1)
        
        notebook = ttk.Notebook(params_frame)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        area_tab = ttk.Frame(notebook, padding=5)
        notebook.add(area_tab, text="Область")
        area_tab.columnconfigure(1, weight=1)
        ttk.Label(area_tab, text="Ширина W (мм)").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.W_var = tk.DoubleVar(value=self.W)
        ttk.Scale(area_tab, from_=100, to=10000, variable=self.W_var, orient=tk.HORIZONTAL, length=180).grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.W_label = ttk.Label(area_tab, text=f"{self.W:.0f} мм", cursor="hand2")
        self.W_label.grid(row=0, column=2, padx=5)
        self.W_label.bind("<Button-1>", lambda event: self.prompt_value("Ширина W (мм)", self.W_var, 100, 10000))
        
        ttk.Label(area_tab, text="Высота H (мм)").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.H_var = tk.DoubleVar(value=self.H)
        ttk.Scale(area_tab, from_=100, to=10000, variable=self.H_var, orient=tk.HORIZONTAL, length=180).grid(row=1, column=1, sticky=(tk.W, tk.E))
        self.H_label = ttk.Label(area_tab, text=f"{self.H:.0f} мм", cursor="hand2")
        self.H_label.grid(row=1, column=2, padx=5)
        self.H_label.bind("<Button-1>", lambda event: self.prompt_value("Высота H (мм)", self.H_var, 100, 10000))
        
        ttk.Label(area_tab, text="Wres (пикс)").grid(row=2, column=0, sticky=tk.W, pady=(10, 2))
        self.Wres_var = tk.IntVar(value=self.Wres)
        ttk.Scale(area_tab, from_=200, to=800, variable=self.Wres_var, orient=tk.HORIZONTAL, length=180).grid(row=2, column=1, sticky=(tk.W, tk.E))
        self.Wres_label = ttk.Label(area_tab, text=f"{self.Wres}", cursor="hand2")
        self.Wres_label.grid(row=2, column=2, padx=5)
        self.Wres_label.bind("<Button-1>", lambda event: self.prompt_value("Разрешение Wres", self.Wres_var, 200, 800, integer=True))
        
        ttk.Label(area_tab, text="Hres (пикс)").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.Hres_var = tk.IntVar(value=self.Hres)
        ttk.Scale(area_tab, from_=200, to=800, variable=self.Hres_var, orient=tk.HORIZONTAL, length=180).grid(row=3, column=1, sticky=(tk.W, tk.E))
        self.Hres_label = ttk.Label(area_tab, text=f"{self.Hres}", cursor="hand2")
        self.Hres_label.grid(row=3, column=2, padx=5)
        self.Hres_label.bind("<Button-1>", lambda event: self.prompt_value("Разрешение Hres", self.Hres_var, 200, 800, integer=True))
        
        light_tab = ttk.Frame(notebook, padding=5)
        notebook.add(light_tab, text="Источник")
        light_tab.columnconfigure(1, weight=1)
        ttk.Label(light_tab, text="xL (мм)").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.xL_var = tk.DoubleVar(value=self.xL)
        ttk.Scale(light_tab, from_=-10000, to=10000, variable=self.xL_var, orient=tk.HORIZONTAL, length=180).grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.xL_label = ttk.Label(light_tab, text=f"{self.xL:.0f} мм", cursor="hand2")
        self.xL_label.grid(row=0, column=2, padx=5)
        self.xL_label.bind("<Button-1>", lambda event: self.prompt_value("Координата xL (мм)", self.xL_var, -10000, 10000))
        
        ttk.Label(light_tab, text="yL (мм)").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.yL_var = tk.DoubleVar(value=self.yL)
        ttk.Scale(light_tab, from_=-10000, to=10000, variable=self.yL_var, orient=tk.HORIZONTAL, length=180).grid(row=1, column=1, sticky=(tk.W, tk.E))
        self.yL_label = ttk.Label(light_tab, text=f"{self.yL:.0f} мм", cursor="hand2")
        self.yL_label.grid(row=1, column=2, padx=5)
        self.yL_label.bind("<Button-1>", lambda event: self.prompt_value("Координата yL (мм)", self.yL_var, -10000, 10000))
        
        ttk.Label(light_tab, text="zL (мм)").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.zL_var = tk.DoubleVar(value=self.zL)
        ttk.Scale(light_tab, from_=100, to=10000, variable=self.zL_var, orient=tk.HORIZONTAL, length=180).grid(row=2, column=1, sticky=(tk.W, tk.E))
        self.zL_label = ttk.Label(light_tab, text=f"{self.zL:.0f} мм", cursor="hand2")
        self.zL_label.grid(row=2, column=2, padx=5)
        self.zL_label.bind("<Button-1>", lambda event: self.prompt_value("Координата zL (мм)", self.zL_var, 100, 10000))
        
        ttk.Label(light_tab, text="I0 (Вт/ср)").grid(row=3, column=0, sticky=tk.W, pady=(10, 2))
        self.I0_var = tk.DoubleVar(value=self.I0)
        ttk.Scale(light_tab, from_=0.01, to=10000, variable=self.I0_var, orient=tk.HORIZONTAL, length=180).grid(row=3, column=1, sticky=(tk.W, tk.E))
        self.I0_label = ttk.Label(light_tab, text=f"{self.I0:.2f} Вт/ср", cursor="hand2")
        self.I0_label.grid(row=3, column=2, padx=5)
        self.I0_label.bind("<Button-1>", lambda event: self.prompt_value("Сила излучения I0 (Вт/ср)", self.I0_var, 0.01, 10000))
        
        circle_tab = ttk.Frame(notebook, padding=5)
        notebook.add(circle_tab, text="Круг")
        circle_tab.columnconfigure(1, weight=1)
        ttk.Label(circle_tab, text="xC (мм)").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.circle_x_var = tk.DoubleVar(value=self.circle_x)
        ttk.Scale(circle_tab, from_=-5000, to=5000, variable=self.circle_x_var, orient=tk.HORIZONTAL, length=180).grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.circle_x_label = ttk.Label(circle_tab, text=f"{self.circle_x:.0f} мм", cursor="hand2")
        self.circle_x_label.grid(row=0, column=2, padx=5)
        self.circle_x_label.bind("<Button-1>", lambda event: self.prompt_value("Центр круга X (мм)", self.circle_x_var, -5000, 5000))
        
        ttk.Label(circle_tab, text="yC (мм)").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.circle_y_var = tk.DoubleVar(value=self.circle_y)
        ttk.Scale(circle_tab, from_=-5000, to=5000, variable=self.circle_y_var, orient=tk.HORIZONTAL, length=180).grid(row=1, column=1, sticky=(tk.W, tk.E))
        self.circle_y_label = ttk.Label(circle_tab, text=f"{self.circle_y:.0f} мм", cursor="hand2")
        self.circle_y_label.grid(row=1, column=2, padx=5)
        self.circle_y_label.bind("<Button-1>", lambda event: self.prompt_value("Центр круга Y (мм)", self.circle_y_var, -5000, 5000))
        
        ttk.Label(circle_tab, text="Радиус R (мм)").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.circle_r_var = tk.DoubleVar(value=self.circle_r)
        ttk.Scale(circle_tab, from_=100, to=5000, variable=self.circle_r_var, orient=tk.HORIZONTAL, length=180).grid(row=2, column=1, sticky=(tk.W, tk.E))
        self.circle_r_label = ttk.Label(circle_tab, text=f"{self.circle_r:.0f} мм", cursor="hand2")
        self.circle_r_label.grid(row=2, column=2, padx=5)
        self.circle_r_label.bind("<Button-1>", lambda event: self.prompt_value("Радиус круга R (мм)", self.circle_r_var, 100, 5000))
        
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=1, column=0, pady=(10, 0), sticky=(tk.W, tk.E))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        
        ttk.Button(button_frame, text="Рассчитать", command=self.calculate).grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Сохранить изображение", command=self.save_image).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))
        
        section_frame = ttk.LabelFrame(main_frame, text="График сечения", padding="5")
        section_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        section_frame.rowconfigure(0, weight=1)
        section_frame.columnconfigure(0, weight=1)
        
        image_frame = ttk.LabelFrame(main_frame, text="Распределение освещенности", padding="5")
        image_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        
        self.setup_slider_callbacks()
        
        self.section_fig = plt.Figure(figsize=(5, 3))
        self.section_canvas = FigureCanvasTkAgg(self.section_fig, section_frame)
        section_widget = self.section_canvas.get_tk_widget()
        section_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        section_widget.configure(height=260)
        
        self.image_fig = plt.Figure(figsize=(5, 3))
        self.image_canvas = FigureCanvasTkAgg(self.image_fig, image_frame)
        image_widget = self.image_canvas.get_tk_widget()
        image_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_widget.configure(height=260)
        
        self.calculate()
        
    # ------------------------------ CALLBACKS ----------------------------- #
        
    def setup_slider_callbacks(self):
        """Обновление подписей у всех параметров при движении ползунков."""
        self.W_var.trace_add("write", lambda *args: self.update_label(self.W_var, self.W_label, " мм"))
        self.H_var.trace_add("write", lambda *args: self.update_label(self.H_var, self.H_label, " мм"))
        self.Wres_var.trace_add("write", lambda *args: self.update_label(self.Wres_var, self.Wres_label, ""))
        self.Hres_var.trace_add("write", lambda *args: self.update_label(self.Hres_var, self.Hres_label, ""))
        self.xL_var.trace_add("write", lambda *args: self.update_label(self.xL_var, self.xL_label, " мм"))
        self.yL_var.trace_add("write", lambda *args: self.update_label(self.yL_var, self.yL_label, " мм"))
        self.zL_var.trace_add("write", lambda *args: self.update_label(self.zL_var, self.zL_label, " мм"))
        self.I0_var.trace_add("write", lambda *args: self.update_label(self.I0_var, self.I0_label, " Вт/ср", format_str="{:.2f}"))
        self.circle_x_var.trace_add("write", lambda *args: self.update_label(self.circle_x_var, self.circle_x_label, " мм"))
        self.circle_y_var.trace_add("write", lambda *args: self.update_label(self.circle_y_var, self.circle_y_label, " мм"))
        self.circle_r_var.trace_add("write", lambda *args: self.update_label(self.circle_r_var, self.circle_r_label, " мм"))
        
    def update_label(self, var, label, suffix="", format_str="{:.0f}"):
        """Перерисовывает подпись рядом со слайдером (используется колбэком trace)."""
        label.config(text=format_str.format(var.get()) + suffix)
    
    def prompt_value(self, title, variable, min_val, max_val, integer=False):
        """Диалог ручного ввода значения параметра; контролирует допустимый диапазон."""
        current_value = variable.get()
        prompt = f"{title}\nДопустимый диапазон: [{min_val}, {max_val}]"
        value = simpledialog.askstring("Ввод значения", prompt, initialvalue=f"{current_value}")
        if value is None:
            return
        try:
            value = float(value.replace(",", "."))
        except ValueError:
            messagebox.showerror("Ошибка ввода", "Введите корректное число.")
            return
        if value < min_val or value > max_val:
            messagebox.showerror("Ошибка ввода", f"Значение должно быть в диапазоне [{min_val}, {max_val}].")
            return
        if integer:
            value = int(round(value))
        variable.set(value)
        
    # ------------------------------ ГЛАВНАЯ ФИЗИЧЕСКАЯ МАТЕМАТИКА ----------------------------- #
        
    def calculate_illumination(self):
        """
        Основная функция расчета двумерного поля освещенности E(x, y).

        Возвращает:
            E — матрица освещенности
            X, Y — координатные сетки
        """
        W = self.W_var.get()
        H = self.H_var.get()
        Wres = self.Wres_var.get()
        Hres = self.Hres_var.get()
        xL = self.xL_var.get()
        yL = self.yL_var.get()
        zL = self.zL_var.get()
        I0 = self.I0_var.get()
        circle_x = self.circle_x_var.get()
        circle_y = self.circle_y_var.get()
        circle_r = self.circle_r_var.get()
        
        # поддерживаем одинаковый физический шаг по X и Y, чтобы пиксели были квадратными
        pixel_size_x = W / Wres
        pixel_size_y = H / Hres
        
        # Если физический размер пикселя по X ≠ по Y → исправляем разрешение
        if abs(pixel_size_x - pixel_size_y) > 0.01:
            pixel_size = max(pixel_size_x, pixel_size_y)
            Wres = int(W / pixel_size)
            Hres = int(H / pixel_size)
            self.Wres_var.set(Wres)
            self.Hres_var.set(Hres)


        # Координаты идут от -W/2 до +W/2, чтобы центр был (0,0)
        x = np.linspace(-W/2, W/2, Wres)
        y = np.linspace(-H/2, H/2, Hres)
        X, Y = np.meshgrid(x, y)
        
        # вектор от каждой точки до источника
        dx = X - xL
        dy = Y - yL
        dz = zL
        
        # расстояние до источника и косинус угла падения для ламбертовского источника
        r_squared = dx**2 + dy**2 + dz**2
        r = np.sqrt(r_squared)
        
        cos_theta = dz / r
        
        # формула E = I0 * cos(theta) / r^2
        E = (I0 * cos_theta) / r_squared
        
        # обнуляем освещенность вне заданного круга
        mask = (X - circle_x)**2 + (Y - circle_y)**2 <= circle_r**2
        E[~mask] = 0
        
        return E, X, Y
        
    # ------------------------------ ОТДЕЛЬНАЯ ТОЧКА ----------------------------- #
    
    def calculate_point_illumination(self, x, y, xL, yL, zL, I0):
        """Расчет освещённости одной точки, используется для отчёта."""
        # Вектор до источника
        dx = x - xL
        dy = y - yL
        dz = zL
        
        r_squared = dx**2 + dy**2 + dz**2
        r = np.sqrt(r_squared)
        
        if r == 0:
            return 0.0
        
        cos_theta = dz / r
        E_mm2 = (I0 * cos_theta) / r_squared
        
        # перевод из мм² в м² для отчета
        E_m2 = E_mm2 * 1e6
        
        return E_m2
    
    # ------------------------------ СТАТИСТИКА ПО КРУГУ ----------------------------- #
    
    def calculate_statistics(self, E, X, Y, circle_x, circle_y, circle_r):
        """max/min/mean внутри круга."""
        # анализируем только значения внутри круга
        mask = (X - circle_x)**2 + (Y - circle_y)**2 <= circle_r**2
        E_in_circle = E[mask]
        
        if len(E_in_circle) == 0:
            return None, None, None
        
        E_max_mm2 = np.max(E_in_circle)
        E_min_mm2 = np.min(E_in_circle)
        E_mean_mm2 = np.mean(E_in_circle)
        
        E_max_m2 = E_max_mm2 * 1e6
        E_min_m2 = E_min_mm2 * 1e6
        E_mean_m2 = E_mean_mm2 * 1e6
        
        return E_max_m2, E_min_m2, E_mean_m2
    
    def update_results_display(self, E, X, Y):
        """Обновляет текстовую панель со значениями в пяти точках и сводной статистикой."""
        circle_x = self.circle_x_var.get()
        circle_y = self.circle_y_var.get()
        circle_r = self.circle_r_var.get()
        xL = self.xL_var.get()
        yL = self.yL_var.get()
        zL = self.zL_var.get()
        I0 = self.I0_var.get()
        
        center_E = self.calculate_point_illumination(circle_x, circle_y, xL, yL, zL, I0)
        
        x_plus_E = self.calculate_point_illumination(circle_x + circle_r, circle_y, xL, yL, zL, I0)
        x_minus_E = self.calculate_point_illumination(circle_x - circle_r, circle_y, xL, yL, zL, I0)
        
        y_plus_E = self.calculate_point_illumination(circle_x, circle_y + circle_r, xL, yL, zL, I0)
        y_minus_E = self.calculate_point_illumination(circle_x, circle_y - circle_r, xL, yL, zL, I0)
        
        E_max, E_min, E_mean = self.calculate_statistics(E, X, Y, circle_x, circle_y, circle_r)
        
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        results = f"Освещенность в точках:\n"
        results += f"{'='*45}\n\n"
        results += f"Центр круга ({circle_x:.1f}, {circle_y:.1f}):\n"
        results += f"  E = {center_E:.6e} Вт/м²\n\n"
        results += f"Пересечение с осью X:\n"
        results += f"  Точка ({circle_x + circle_r:.1f}, {circle_y:.1f}):\n"
        results += f"    E = {x_plus_E:.6e} Вт/м²\n"
        results += f"  Точка ({circle_x - circle_r:.1f}, {circle_y:.1f}):\n"
        results += f"    E = {x_minus_E:.6e} Вт/м²\n\n"
        results += f"Пересечение с осью Y:\n"
        results += f"  Точка ({circle_x:.1f}, {circle_y + circle_r:.1f}):\n"
        results += f"    E = {y_plus_E:.6e} Вт/м²\n"
        results += f"  Точка ({circle_x:.1f}, {circle_y - circle_r:.1f}):\n"
        results += f"    E = {y_minus_E:.6e} Вт/м²\n\n"
        results += f"{'='*45}\n"
        results += f"Статистика в пределах круга:\n\n"
        if E_max is not None:
            results += f"Максимальное значение:\n"
            results += f"  E_max = {E_max:.6e} Вт/м²\n\n"
            results += f"Минимальное значение:\n"
            results += f"  E_min = {E_min:.6e} Вт/м²\n\n"
            results += f"Среднее значение:\n"
            results += f"  E_mean = {E_mean:.6e} Вт/м²\n"
        else:
            results += f"  Нет данных в пределах круга\n"
        
        self.results_text.insert(1.0, results)
        self.results_text.config(state=tk.DISABLED)
        
    def calculate(self):
        """Общий цикл: расчёт → нормировка → графики → таблица значений."""
        E, X, Y = self.calculate_illumination()
        
        # нормировка карты освещенности в диапазон [0, 255] для визуализации и сохранения
        E_normalized = np.zeros_like(E)
        E_max = np.max(E)
        if E_max > 0:
            E_normalized = (E / E_max * 255).astype(np.uint8)
        
        self.illumination_image = E_normalized
        self.E_data = E
        self.X_data = X
        self.Y_data = Y
        
        self.update_results_display(E, X, Y)
        
        self.image_fig.clear()
        ax_img = self.image_fig.add_subplot(1, 1, 1)
        im = ax_img.imshow(
            E_normalized,
            cmap='gray',
            origin='lower',
            extent=[X.min(), X.max(), Y.min(), Y.max()],
        )
        ax_img.set_xlabel('X (мм)')
        ax_img.set_ylabel('Y (мм)')
        ax_img.set_title('Распределение освещенности')
        self.image_fig.colorbar(im, ax=ax_img, label='Освещенность (нормированная 0-255)', fraction=0.046, pad=0.04)
        self.image_fig.tight_layout()
        self.image_canvas.draw()
        
        self.section_fig.clear()
        ax_sec = self.section_fig.add_subplot(1, 1, 1)
        # строим профили вдоль центральных осей
        center_x_idx = self.E_data.shape[1] // 2
        center_y_idx = self.E_data.shape[0] // 2
        
        horizontal_section = self.E_data[center_y_idx, :]
        vertical_section = self.E_data[:, center_x_idx]
        
        x_coords = X[center_y_idx, :]
        y_coords = Y[:, center_x_idx]
        
        ax_sec.plot(x_coords, horizontal_section, label='Горизонтальное сечение', linewidth=2)
        ax_sec.plot(y_coords, vertical_section, label='Вертикальное сечение', linewidth=2)
        ax_sec.set_xlabel('Координата (мм)')
        ax_sec.set_ylabel('Освещенность')
        ax_sec.set_title('Сечения через центр области')
        ax_sec.legend()
        ax_sec.grid(True, alpha=0.3)
        self.section_fig.tight_layout()
        self.section_canvas.draw()
        
    # ------------------------------ СОХРАНЕНИЕ ----------------------------- #
        
    def save_image(self):
        """Сохраняет нормированную карту освещённости."""
        if not hasattr(self, 'illumination_image'):
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if filename:
            img = Image.fromarray(self.illumination_image, mode='L')
            img.save(filename)
            print(f"Изображение сохранено: {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = IlluminationCalculator(root)
    root.mainloop()
