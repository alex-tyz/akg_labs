import numpy as np  # Библиотека численных вычислений
from PIL import Image, ImageTk  # Для работы с изображениями
import tkinter as tk  # Tkinter — стандартная GUI-библиотека Python
from tkinter import ttk, messagebox, filedialog  # Красивые виджеты и диалоги


# =====================================================================
# =============== БАЗОВЫЕ ВЫЧИСЛЕНИЯ ОСВЕЩЕНИЯ СФЕРЫ ===================
# =====================================================================

def render_sphere(
        W, H, Wres, Hres,
        zO,
        z_scr,
        R, Cx, Cy, Cz,
        kd, ks, shininess,
        lights
):
    """
    Основная функция, выполняющая расчёт:
    • пересечения лучей со сферой
    • нормалей
    • освещения по модели Блинна–Фонга
    • формирование итогового изображения

    Возвращает:
    - img_uint8 — изображение (0–255), готовое для отображения
    - I_float   — абсолютные яркости (ненормированные)
    - I_max     — максимальная яркость
    - I_min     — минимальная ненулевая яркость
    """

    # Создаём вектор наблюдателя O = (0, 0, zO)
    O = np.array([0.0, 0.0, zO])

    # Создаём вектор центра сферы
    C = np.array([Cx, Cy, Cz])

    # --------------------------------------------------------------
    # 1. ФОРМИРОВАНИЕ СЕТКИ ПИКСЕЛЕЙ ЭКРАНА
    # --------------------------------------------------------------

    # Создаём координаты пикселей по X: от -W/2 до +W/2
    xs = np.linspace(-W / 2, W / 2, Wres, endpoint=False) + W / (2 * Wres)

    # Аналогично по Y: от -H/2 до +H/2
    ys = np.linspace(-H / 2, H / 2, Hres, endpoint=False) + H / (2 * Hres)

    # 2D-сетка координат всех пикселей (размер: Hres × Wres)
    X, Y = np.meshgrid(xs, ys)

    # Превращаем в массив точек (каждая строка — пиксель)
    Px = X.ravel()
    Py = Y.ravel()
    Pz = np.full_like(Px, z_scr)  # z-координата всех пикселей одинаковая
    P_screen = np.stack([Px, Py, Pz], axis=1)

    # --------------------------------------------------------------
    # 2. ПОСТРОЕНИЕ ЛУЧЕЙ ОТ НАБЛЮДАТЕЛЯ К ПИКСЕЛЯМ
    # --------------------------------------------------------------

    # Вектор направления: от наблюдателя до пикселя
    dirs = P_screen - O

    # Нормировка направлений
    dir_norm = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.maximum(dir_norm, 1e-8)

    # --------------------------------------------------------------
    # 3. РАСЧЁТ ПЕРЕСЕЧЕНИЯ ЛУЧА СО СФЕРОЙ
    # --------------------------------------------------------------

    # Вычисляем коэффициенты квадратного уравнения
    OC = O - C
    a = np.sum(dirs * dirs, axis=1)
    b = 2.0 * np.sum(dirs * OC, axis=1)
    c = np.dot(OC, OC) - R ** 2

    # Дискриминант
    discriminant = b ** 2 - 4 * a * c
    hit_mask = discriminant >= 0.0  # True — луч пересекает сферу

    # Изначально t = ∞ (нет пересечения)
    t = np.full_like(a, np.inf)

    sqrtD = np.zeros_like(a)
    sqrtD[hit_mask] = np.sqrt(discriminant[hit_mask])

    # Два корня квадр. уравнения
    t1 = (-b - sqrtD) / (2 * a)
    t2 = (-b + sqrtD) / (2 * a)

    # Выбираем ближний положительный корень
    t_candidate = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
    positive = (t_candidate > 0) & hit_mask
    t[positive] = t_candidate[positive]

    final_hit_mask = np.isfinite(t)  # True — пиксель действительно видит сферу

    # --------------------------------------------------------------
    # 4. КООРДИНАТЫ ТОЧЕК НА СФЕРЕ
    # --------------------------------------------------------------

    P = O + dirs * t[:, np.newaxis]  # Точки пересечения лучей со сферой

    # --------------------------------------------------------------
    # 5. НОРМАЛИ И ВЕКТОР НАБЛЮДАТЕЛЯ
    # --------------------------------------------------------------

    # Нормаль к сфере N = нормированный (P - C)
    N = P - C
    N_norm = np.linalg.norm(N, axis=1, keepdims=True)
    N = N / np.maximum(N_norm, 1e-8)

    # Вектор к наблюдателю
    V = O - P
    V_norm = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.maximum(V_norm, 1e-8)

    # --------------------------------------------------------------
    # 6. ОСВЕЩЕНИЕ ПО МОДЕЛИ БЛИННА–ФОНГА
    # --------------------------------------------------------------

    I = np.zeros(P.shape[0], dtype=np.float64)

    # Перебираем все источники света
    for (lx, ly, lz, I0) in lights:
        # Позиция источника
        Lpos = np.array([lx, ly, lz])

        # Направление света L_dir
        L = Lpos - P
        L_norm = np.linalg.norm(L, axis=1, keepdims=True)
        L_dir = L / np.maximum(L_norm, 1e-8)

        # Диффузная компонента — Ламберт
        cos_theta = np.sum(N * L_dir, axis=1)
        cos_theta = np.clip(cos_theta, 0.0, None)

        # Полувектор для блика
        Hvec = L_dir + V
        H_norm = np.linalg.norm(Hvec, axis=1, keepdims=True)
        H_dir = Hvec / np.maximum(H_norm, 1e-8)

        # Зеркальная составляющая
        cos_alpha = np.sum(N * H_dir, axis=1)
        cos_alpha = np.clip(cos_alpha, 0.0, None)

        # Диффузная и зеркальная яркость
        I_diff = kd * I0 * cos_theta
        I_spec = ks * I0 * (cos_alpha ** shininess)

        # Общая яркость = сумма от всех источников
        I += I_diff + I_spec

    # Пиксели вне сферы = чёрные
    I[~final_hit_mask] = 0.0

    # Переводим в матрицу Hres × Wres
    I_img = I.reshape(Hres, Wres)

    # Максимальная яркость
    I_max = float(I_img.max())

    # Минимальная ненулевая яркость
    if np.any(I_img > 0):
        I_min = float(I_img[I_img > 0].min())
    else:
        I_min = 0.0

    # --------------------------------------------------------------
    # 7. НОРМИРОВКА 0–255 ДЛЯ ИЗОБРАЖЕНИЯ
    # --------------------------------------------------------------

    if I_max > 0:
        I_norm = (I_img / I_max) * 255.0
    else:
        I_norm = I_img

    img_uint8 = np.clip(I_norm, 0, 255).astype(np.uint8)

    # Возвращаем промежуточные и финальные данные
    return img_uint8, I_img, I_max, I_min


def compute_point_brightness(point, center, observer, kd, ks, shininess, lights):
    """Возвращает абсолютную яркость выбранной точки сферы."""
    normal = point - center
    normal_norm = np.linalg.norm(normal)
    if normal_norm <= 1e-8:
        return 0.0
    normal = normal / normal_norm

    view_vec = observer - point
    view_norm = np.linalg.norm(view_vec)
    if view_norm <= 1e-8:
        return 0.0
    view_dir = view_vec / view_norm

    if np.dot(normal, view_dir) <= 0:
        return 0.0

    intensity = 0.0
    for lx, ly, lz, I0 in lights:
        light_pos = np.array([lx, ly, lz])
        light_vec = light_pos - point
        light_norm = np.linalg.norm(light_vec)
        if light_norm <= 1e-8:
            continue
        light_dir = light_vec / light_norm
        cos_theta = np.dot(normal, light_dir)
        if cos_theta <= 0:
            continue
        diffuse = kd * I0 * cos_theta
        half_vec = light_dir + view_dir
        half_norm = np.linalg.norm(half_vec)
        specular = 0.0
        if half_norm > 1e-8:
            half_vec /= half_norm
            specular = ks * I0 * max(np.dot(normal, half_vec), 0.0) ** shininess
        intensity += diffuse + specular
    return intensity

# =====================================================================
# ============================= GUI ЧАСТЬ ==============================
# =====================================================================

class SphereApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Sphere Lighting (Блинн-Фонг)")  # Заголовок окна
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.style.configure("Param.Horizontal.TScale", troughcolor="#444", background="#f5f5f5", thickness=10)
        self.style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        self.style.configure("Secondary.TButton", font=("Segoe UI", 10), padding=6)

        # Виджет для отображения изображения слева
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, rowspan=20, padx=5, pady=5)

        # Правая панель для настроек
        control_panel = ttk.Frame(self)
        control_panel.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=5)
        control_panel.rowconfigure(0, weight=1)
        control_panel.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(control_panel)
        notebook.grid(row=0, column=0, sticky="nsew")

        def slider_factory(frame):
            frame.columnconfigure(1, weight=1)
            row_state = {"value": 0}

            def add_slider(text, default, min_val, max_val, fmt="{:.0f}", integer=False):
                r = row_state["value"]
                ttk.Label(frame, text=text).grid(row=r, column=0, sticky="w", padx=(0, 4), pady=2)
                var = tk.DoubleVar(value=default)
                scale = ttk.Scale(
                    frame,
                    from_=min_val,
                    to=max_val,
                    variable=var,
                    orient=tk.HORIZONTAL,
                    length=150,
                    style="Param.Horizontal.TScale",
                )
                scale.grid(row=r, column=1, sticky="we", padx=4, pady=2)
                value_lbl = ttk.Label(frame, text=fmt.format(default), width=8, anchor="w")
                value_lbl.grid(row=r, column=2, sticky="w", padx=(2, 0))

                def update_label(*_):
                    value = var.get()
                    if integer:
                        value = round(value)
                    value_lbl.config(text=fmt.format(value))

                var.trace_add("write", update_label)
                row_state["value"] += 1
                return var

            add_slider.row_state = row_state
            return add_slider

        # Tabs: Screen, Observer, Sphere, Material, Lights 1&2
        screen_tab = ttk.Frame(notebook, padding=6)
        observer_tab = ttk.Frame(notebook, padding=6)
        sphere_tab = ttk.Frame(notebook, padding=6)
        material_tab = ttk.Frame(notebook, padding=6)
        light1_tab = ttk.Frame(notebook, padding=6)
        light2_tab = ttk.Frame(notebook, padding=6)

        notebook.add(screen_tab, text="Экран")
        notebook.add(observer_tab, text="Наблюдатель")
        notebook.add(sphere_tab, text="Сфера")
        notebook.add(material_tab, text="Материал")
        notebook.add(light1_tab, text="Источник 1")
        notebook.add(light2_tab, text="Источник 2")

        add_screen = slider_factory(screen_tab)
        add_observer = slider_factory(observer_tab)
        add_sphere = slider_factory(sphere_tab)
        add_material = slider_factory(material_tab)
        add_light1 = slider_factory(light1_tab)
        add_light2 = slider_factory(light2_tab)

        self.W_var = add_screen("W (мм)", 800, 100, 10000)
        self.H_var = add_screen("H (мм)", 600, 100, 10000)
        self.Wres_var = add_screen("Wres (пикс)", 600, 200, 800, fmt="{:.0f}", integer=True)
        hres_row = add_screen.row_state["value"]
        ttk.Label(screen_tab, text="Hres (пикс)").grid(row=hres_row, column=0, sticky="w", pady=2)
        self.Hres_var = tk.StringVar(value="450")
        ttk.Label(screen_tab, textvariable=self.Hres_var).grid(row=hres_row, column=1, columnspan=2, sticky="w", pady=2)
        add_screen.row_state["value"] += 1
        self.zscr_var = add_screen("z_scr (мм)", 0, -5000, 5000)

        self.zO_var = add_observer("zO (мм)", -1000, -10000, 10000)

        self.R_var = add_sphere("R (мм)", 300, 100, 5000)
        self.Cx_var = add_sphere("Cx (мм)", 0, -10000, 10000)
        self.Cy_var = add_sphere("Cy (мм)", 0, -10000, 10000)
        self.Cz_var = add_sphere("Cz (мм)", 800, 100, 10000)

        self.kd_var = add_material("kd", 0.9, 0.0, 1.0, fmt="{:.2f}")
        self.ks_var = add_material("ks", 0.8, 0.0, 1.0, fmt="{:.2f}")
        self.shn_var = add_material("shininess", 50, 1, 200, fmt="{:.0f}", integer=True)

        self.L1x_var = add_light1("L1x", 3000, -10000, 10000)
        self.L1y_var = add_light1("L1y", 2000, -10000, 10000)
        self.L1z_var = add_light1("L1z", -500, -5000, 10000)
        self.I1_var = add_light1("I01", 500, 0.01, 10000, fmt="{:.1f}")

        self.L2x_var = add_light2("L2x", -2000, -10000, 10000)
        self.L2y_var = add_light2("L2y", 3000, -10000, 10000)
        self.L2z_var = add_light2("L2z", -800, -5000, 10000)
        self.I2_var = add_light2("I02", 200, 0.01, 10000, fmt="{:.1f}")

        # Строка вывода максимальной/минимальной яркости
        self.info_var = tk.StringVar(value="")
        ttk.Label(control_panel, textvariable=self.info_var, foreground="#004b8d").grid(
            row=1, column=0, sticky="w", pady=(8, 4)
        )

        results_frame = ttk.LabelFrame(control_panel, text="Расчетные значения", padding=6)
        results_frame.grid(row=2, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        self.results_text = tk.Text(results_frame, height=7, width=36, font=("Courier New", 9), wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky="nsew")
        self.results_text.config(state=tk.DISABLED)

        # Кнопка запуска рендера
        btn = ttk.Button(control_panel, text="Render", command=self.on_render, style="Accent.TButton")
        btn.grid(row=3, column=0, pady=4, sticky="we")

        # Кнопка сохранения изображения
        self.save_btn = ttk.Button(control_panel, text="Save image", command=self.on_save, style="Secondary.TButton")
        self.save_btn.grid(row=4, column=0, pady=(0, 6), sticky="we")

        # Текущие изображения
        self.current_photo = None
        self.last_pil_image = None

        # Первый автоматический рендер
        self.on_render()

    # ----------------------------------------------------------------------
    # ВЫПОЛНЕНИЕ РАСЧЁТА И ПОКАЗ ИЗОБРАЖЕНИЯ
    # ----------------------------------------------------------------------
    def on_render(self):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())

            # Вычисляем Hres так, чтобы пиксели были квадратными:
            # pixel_size = W / Wres = H / Hres  =>  Hres = H / (W / Wres) = Wres * H / W
            if W <= 0:
                raise ValueError("W must be > 0")
            Hres = max(1, int(round(Wres * H / W)))
            self.Hres_var.set(str(Hres))

            z_scr = float(self.zscr_var.get())
            zO = float(self.zO_var.get())

            R = float(self.R_var.get())
            Cx = float(self.Cx_var.get())
            Cy = float(self.Cy_var.get())
            Cz = float(self.Cz_var.get())

            kd = float(self.kd_var.get())
            ks = float(self.ks_var.get())
            sh = float(self.shn_var.get())

            L1x = float(self.L1x_var.get())
            L1y = float(self.L1y_var.get())
            L1z = float(self.L1z_var.get())
            I01 = float(self.I1_var.get())

            L2x = float(self.L2x_var.get())
            L2y = float(self.L2y_var.get())
            L2z = float(self.L2z_var.get())
            I02 = float(self.I2_var.get())

            lights = [
                (L1x, L1y, L1z, I01),
                (L2x, L2y, L2z, I02),
            ]
            observer = np.array([0.0, 0.0, zO])
            center = np.array([Cx, Cy, Cz])

            # Запуск расчёта яркости
            img_uint8, I_float, I_max, I_min = render_sphere(
                W, H, Wres, Hres,
                zO,
                z_scr,
                R, Cx, Cy, Cz,
                kd, ks, sh,
                lights
            )

            # Конвертируем NumPy → PIL
            pil_img = Image.fromarray(img_uint8, mode="L")
            self.last_pil_image = pil_img

            # Показываем изображение (масштабируем, если нужно)
            display_img = pil_img
            self.current_photo = ImageTk.PhotoImage(display_img)
            self.image_label.configure(image=self.current_photo)

            # Показываем максимум и минимум яркости
            self.info_var.set(f"Max = {I_max:.3g}, Min>0 = {I_min:.3g}")
            sample_points = self.get_sample_points(center, observer, R)
            point_values = [
                (label, point, compute_point_brightness(point, center, observer, kd, ks, sh, lights))
                for label, point in sample_points
            ]
            self.update_results_panel(point_values, I_max, I_min)

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    # ----------------------------------------------------------------------
    # СОХРАНЕНИЕ ИЗОБРАЖЕНИЯ В ФАЙЛ
    # ----------------------------------------------------------------------
    def on_save(self):
        """Сохранение текущего изображения."""
        if self.last_pil_image is None:
            messagebox.showwarning("Нет изображения", "Сначала нажмите Render.")
            return

        filename = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("All files", "*.*")]
        )

        if filename:
            try:
                self.last_pil_image.save(filename)
                messagebox.showinfo("Сохранено", f"Изображение сохранено в:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка сохранения", str(e))

    def get_sample_points(self, center, observer, radius):
        """Возвращает три контрольные точки на сфере."""
        front_dir = observer - center
        norm = np.linalg.norm(front_dir)
        if norm <= 1e-8:
            front_dir = np.array([0.0, 0.0, 1.0])
        else:
            front_dir = front_dir / norm
        front_point = center + front_dir * radius
        x_point = center + np.array([radius, 0.0, 0.0])
        y_point = center + np.array([0.0, radius, 0.0])
        return [
            ("Видимая точка", front_point),
            ("+X направление", x_point),
            ("+Y направление", y_point),
        ]

    def update_results_panel(self, point_values, max_val, min_val):
        """Выводит расчетные значения яркости в текстовом окне."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        header = "Точка               L (Вт/м²)\n"
        header += "-" * 32 + "\n"
        self.results_text.insert(tk.END, header)
        for label, point, value in point_values:
            self.results_text.insert(
                tk.END,
                f"{label:16s}: {value: .5e}\n"
            )
        self.results_text.insert(tk.END, "\n")
        self.results_text.insert(tk.END, f"Lmax = {max_val:.5e}\n")
        self.results_text.insert(tk.END, f"Lmin = {min_val:.5e}\n")
        self.results_text.config(state=tk.DISABLED)


# =====================================================================
# ======================== ЗАПУСК ПРИЛОЖЕНИЯ ==========================
# =====================================================================
if __name__ == "__main__":
    app = SphereApp()  # создаём объект GUI
    app.mainloop()  # запускаем главный цикл Tkinter
