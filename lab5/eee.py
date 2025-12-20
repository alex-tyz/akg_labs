import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser


# =====================================================================
# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =======================
# =====================================================================

# --- Пересечение луча и сферы ---
def ray_sphere_intersect(O, D, C, R):
    """
    Находит ближайшее положительное пересечение луча O + t*D со сферой (C, R).
    Возвращает: t (массив), hit_mask (boolean)
    """
    OC = O - C
    a = np.sum(D * D, axis=-1)
    b = 2.0 * np.sum(D * OC, axis=-1)
    c = np.sum(OC * OC, axis=-1) - R * R
    disc = b * b - 4 * a * c
    hit = disc >= 0
    t = np.full_like(a, np.inf)
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denom = 2 * a + 1e-8
    t1 = (-b - sqrt_disc) / denom
    t2 = (-b + sqrt_disc) / denom
    t_candidate = np.where((t1 > 0) & ((t1 < t2) | (t2 <= 0)), t1, t2)
    t = np.where((t_candidate > 0) & hit, t_candidate, np.inf)
    return t, t != np.inf

# --- Основной рендер сцены ---
def render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights):
    """
    Трассирует лучи от наблюдателя к экрану и рассчитывает освещённость
    по модели Блинна–Фонга с проверкой теней.
    """
    O = np.array([0.0, 0.0, zO], dtype=np.float64)

    # Сетка центров пикселей на экране; фактически дискретизируем прямоугольный экран
    # так, чтобы лучи шли через центр каждого пикселя. Инверсия оси Y нужна, чтобы
    # «верх» на экране соответствовал положительному направлению мировой оси Y.
    xs = np.linspace(-W / 2, W / 2, Wres, endpoint=False) + W / (2 * Wres)
    ys = np.linspace(H / 2, -H / 2, Hres, endpoint=False) - H / (2 * Hres)

    X, Y = np.meshgrid(xs, ys)
    Px = X.ravel()
    Py = Y.ravel()
    Pz = np.full_like(Px, z_scr)
    P_screen = np.stack([Px, Py, Pz], axis=1)

    D = P_screen - O
    D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D = D / np.maximum(D_norm, 1e-8)  # нормализуем, чтобы t измерялось в миллиметрах

    depth = np.full(P_screen.shape[0], np.inf, dtype=np.float64)
    sphere_id = np.full(P_screen.shape[0], -1, dtype=int)

    for i, sph in enumerate(spheres):
        C = np.array(sph['center'], dtype=np.float64)
        R = sph['radius']
        # Для каждой сферы решаем квадратное уравнение пересечения; получаем массив t
        t, hit = ray_sphere_intersect(O, D, C, R)  # решаем |O + tD - C|^2 = R^2
        closer = t < depth
        mask = hit & closer
        depth[mask] = t[mask]
        sphere_id[mask] = i

    valid = sphere_id >= 0
    valid_idx = np.where(valid)[0]

    P = np.zeros_like(D)
    if valid_idx.size == 0:
        img_rgb = np.zeros((Hres, Wres, 3))
        return (np.zeros((Hres, Wres, 3), dtype=np.uint8), img_rgb, 0.0, 0.0)

    P[valid_idx] = O + D[valid_idx] * depth[valid_idx, np.newaxis]  # точки на поверхности

    V = np.zeros_like(D)
    V[valid_idx] = O - P[valid_idx]  # обратные направления к наблюдателю
    V_norm = np.linalg.norm(V[valid_idx], axis=1, keepdims=True)
    V[valid_idx] = V[valid_idx] / np.maximum(V_norm, 1e-8)

    final_color = np.zeros((P_screen.shape[0], 3), dtype=np.float64)

    N = np.zeros_like(P)
    # Для каждого пикселя храним центр соответствующей сферы и строим нормаль
    centers_arr = np.array([spheres[i]['center']
                            for i in sphere_id[valid_idx]], dtype=np.float64)
    N[valid_idx] = P[valid_idx] - centers_arr
    N_norm = np.linalg.norm(N[valid_idx], axis=1, keepdims=True)
    N[valid_idx] /= np.maximum(N_norm, 1e-8)

    for idx in valid_idx:
        sph = spheres[sphere_id[idx]]
        point = P[idx]
        normal = N[idx]
        color_acc = np.zeros(3)

        for light in lights:
            L_pos = np.array(light['pos'], dtype=np.float64)
            L_dir_vec = L_pos - point
            L_dist = np.linalg.norm(L_dir_vec)
            if L_dist < 1e-6:
                continue
            L_dir = L_dir_vec / L_dist  # единичный вектор на источник
            light_color = np.array(light['color'], dtype=np.float64)
            I0 = light['I0']

            shadow_ray_origin = point + normal * 1e-2  # Чуть больший сдвиг, чтобы избежать шума
            shadow = False
            for other_sph in spheres:  # трассируем «теневой» луч
                C_other = np.array(other_sph['center'], dtype=np.float64)
                R_other = other_sph['radius']
                t_shadow, hit_shadow = ray_sphere_intersect(
                    shadow_ray_origin, L_dir, C_other, R_other)
                if np.any(hit_shadow) and np.min(t_shadow[hit_shadow]) < L_dist - 1e-2:
                    shadow = True
                    break

            if shadow:
                continue  # источник полностью перекрыт — нет вклада

            diff = max(np.dot(normal, L_dir), 0.0)  # диффузная ламбертовская компонента
            I_diff = sph['kd'] * I0 * diff

            H_vec = L_dir + V[idx]  # полувектор Блинна
            H_norm_val = np.linalg.norm(H_vec)
            if H_norm_val < 1e-8:
                spec = 0.0
            else:
                H_vec = H_vec / H_norm_val
                spec = max(np.dot(normal, H_vec), 0.0)
                spec = spec ** sph['shininess']
            I_spec = sph['ks'] * I0 * spec  # зеркальная составляющая

            contrib = (I_diff + I_spec)
            color_acc += contrib * light_color * np.array(sph['color'])

        final_color[idx] = color_acc  # RGB вклад для пикселя

    # Формируем изображение после обработки всех пикселей
    img_rgb = final_color.reshape((Hres, Wres, 3))

    I_max = img_rgb.max()
    I_min = 0.0
    if I_max > 0:
        non_zero = img_rgb[img_rgb > 0]
        if len(non_zero) > 0:
            I_min = non_zero.min()

    if I_max > 0:
        img_norm = (img_rgb / I_max) * 255
    else:
        img_norm = img_rgb
    img_uint8 = np.clip(img_norm, 0, 255).astype(np.uint8)

    return img_uint8, img_rgb, I_max, I_min

# --- Построение ортогональных проекций ---
def render_projections(W, H, Wres, Hres, spheres, lights):
    """
    Рендерит сцену в трёх ортогональных видах (XY/XZ/YZ),
    переставляя координаты сфер и источников.
    """
    projections = {}

    # Камера для ортогональных проекций ставится далеко
    zO_proj = 5000.0
    z_scr = 0.0

    # 1. Фронтальная проекция (XY) — классический вид спереди:
    # камера смотрит вдоль оси Z, поэтому координаты объектов менять не нужно.
    spheres_xy = spheres
    lights_xy = lights

    img_xy, rgb_xy, imax_xy, imin_xy = render_scene(
        W, H, Wres, Hres, zO_proj, z_scr, spheres_xy, lights_xy
    )
    projections['frontal'] = {
        'image': img_xy,
        'rgb': rgb_xy,
        'I_max': imax_xy,
        'I_min': imin_xy,
        'name': 'Вид XY (Z)',
        'description': 'Фронтальная проекция'
    }

    # 2. Горизонтальная проекция (XZ) — вид сверху: камера смотрит вдоль оси Y.
    # Экран показывает X по горизонтали и Z по вертикали, а бывшая ось Y становится глубиной.
    spheres_top = []
    for sph in spheres:
        cx, cy, cz = sph['center']
        spheres_top.append({**sph, 'center': (cx, cz, cy)})  # Z переносим в «вертикаль»

    lights_top = []
    for light in lights:
        lx, ly, lz = light['pos']
        lights_top.append({**light, 'pos': (lx, lz, ly)})

    img_top, rgb_top, imax_top, imin_top = render_scene(
        W, H, Wres, Hres, zO_proj, z_scr, spheres_top, lights_top
    )
    projections['horizontal'] = {
        'image': img_top,
        'rgb': rgb_top,
        'I_max': imax_top,
        'I_min': imin_top,
        'name': 'Вид XZ (Y)',
        'description': 'Горизонтальная проекция'
    }

    # 3. Профильная проекция (YZ) — вид сбоку: камера вдоль X, поэтому горизонталь экрана = Y.
    spheres_side = []
    for sph in spheres:
        cx, cy, cz = sph['center']
        spheres_side.append({**sph, 'center': (cy, cz, cx)})  # Y становится горизонталью

    lights_side = []
    for light in lights:
        lx, ly, lz = light['pos']
        lights_side.append({**light, 'pos': (ly, lz, lx)})

    img_side, rgb_side, imax_side, imin_side = render_scene(
        W, H, Wres, Hres, zO_proj, z_scr, spheres_side, lights_side
    )
    projections['profile'] = {
        'image': img_side,
        'rgb': rgb_side,
        'I_max': imax_side,
        'I_min': imin_side,
        'name': 'Вид YZ (X)',
        'description': 'Профильная проекция'
    }

    return projections


# =====================================================================
# ====================== GUI (ВАШ ДИЗАЙН) =============================
# =====================================================================

class ModernSceneApp(tk.Tk):
    # --- Конструктор приложения: настройка GUI и запуск первого рендера ---
    def __init__(self):
        """Создаёт главное окно приложения, настраивает стиль и компоновку."""
        super().__init__()
        self.title("ЛР5: Визуализация сфер с тенями (Блинн-Фонг)")
        self.geometry("1400x850")
        self.configure(bg='#f5f5f5')

        # Стили - пастельные тона
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#f5f5f5')
        style.configure('TLabel', background='#f5f5f5',
                        foreground='#4a4a4a', font=('Segoe UI', 9))
        style.configure('TLabelframe', background='#f5f5f5',
                        foreground='#8b7fa8', borderwidth=2)
        style.configure('TLabelframe.Label', background='#f5f5f5',
                        foreground='#8b7fa8', font=('Segoe UI', 10, 'bold'))
        style.configure('TButton', background='#bae1ff', foreground='#2c3e50',
                        borderwidth=0, font=('Segoe UI', 10, 'bold'))
        style.map('TButton', background=[('active', '#9dd1ff')])
        style.configure('TEntry', fieldbackground='#ffffff',
                        foreground='#2c3e50', borderwidth=1)
        style.configure('TNotebook', background='#f5f5f5', borderwidth=0)
        style.configure('TNotebook.Tab', background='#e8e8e8',
                        foreground='#4a4a4a', padding=[15, 5])
        style.map('TNotebook.Tab', background=[
            ('selected', '#bae1ff')], foreground=[('selected', '#2c3e50')])

        # Основной контейнер
        main_container = ttk.Frame(self)
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Левая панель: превью, кнопки и проекции
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        self.image_label = tk.Label(
            left_panel, bg='black', relief='sunken', bd=2)
        self.image_label.pack(fill=tk.BOTH, expand=True)

        # Информационная панель
        info_frame = ttk.Frame(left_panel)
        info_frame.pack(fill=tk.X, pady=(10, 0))

        self.info_var = tk.StringVar(value="Яркость: Max = 0.0, Min = 0.0")
        info_label = ttk.Label(info_frame, textvariable=self.info_var, font=(
            'Segoe UI', 10), foreground='#6b8e6b')
        info_label.pack(anchor='w')

        # Кнопки действий
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        self.render_btn = ttk.Button(
            btn_frame, text="РЕНДЕРИТЬ", command=self.render)
        self.render_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=20, ipady=8)

        self.save_btn = ttk.Button(
            btn_frame, text="СОХРАНИТЬ", command=self.save)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 10), ipadx=20, ipady=8)

        self.save_proj_btn = ttk.Button(
            btn_frame, text="СОХР. ПРОЕКЦИИ", command=self.save_projections)
        self.save_proj_btn.pack(side=tk.LEFT, ipadx=20, ipady=8)

        projections_frame = ttk.LabelFrame(left_panel, text="Ортогональные проекции", padding=10)
        projections_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        for col in range(3):
            projections_frame.columnconfigure(col, weight=1)

        self.projection_labels = {}
        self.projection_images = {}
        projection_titles = [
            ('frontal', 'Фронтальная (XY)'),
            ('horizontal', 'Горизонтальная (XZ)'),
            ('profile', 'Профильная (YZ)')
        ]
        for col, (key, title) in enumerate(projection_titles):
            cell = ttk.Frame(projections_frame)
            cell.grid(row=0, column=col, padx=5, sticky='nsew')
            ttk.Label(cell, text=title, anchor='center').pack(fill=tk.X)
            lbl = tk.Label(cell, bg='black', fg='white', text='Нет данных', relief='sunken', bd=1)
            lbl.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
            self.projection_labels[key] = lbl
            self.projection_images[key] = None

        # Правая панель - скроллируемый список параметров сцены
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        params_label = ttk.Label(right_panel, text="ПАРАМЕТРЫ", font=(
            'Segoe UI', 14, 'bold'), foreground='#8b7fa8')
        params_label.pack(pady=(0, 10))

        params_canvas = tk.Canvas(right_panel, bg='#f5f5f5', highlightthickness=0)
        params_scroll = ttk.Scrollbar(right_panel, orient=tk.VERTICAL, command=params_canvas.yview)
        params_canvas.configure(yscrollcommand=params_scroll.set)
        params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        params_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        params_frame = ttk.Frame(params_canvas)
        params_canvas.create_window((0, 0), window=params_frame, anchor='nw')

        def update_params_scroll(event):
            params_canvas.configure(scrollregion=params_canvas.bbox("all"))

        params_frame.bind("<Configure>", update_params_scroll)

        self._build_camera_section(params_frame)
        self._sync_lock = False
        self.W_var.trace_add("write", self._on_WH_change)
        self.H_var.trace_add("write", self._on_WH_change)
        self.Wres_var.trace_add("write", self._on_Wres_change)
        self.Hres_var.trace_add("write", self._on_Hres_change)

        sphere_defaults = [
            {"title": "Сфера 1", "radius": 200, "center": (-150, 0, 0), "kd": 0.7, "ks": 0.3,
             "shininess": 30, "color": (0.2, 0.4, 1.0)},
            {"title": "Сфера 2", "radius": 120, "center": (200, 0, 0), "kd": 0.6, "ks": 0.4,
             "shininess": 50, "color": (1.0, 0.9, 0.2)}
        ]
        for idx, defaults in enumerate(sphere_defaults, start=1):
            self._build_sphere_section(params_frame, idx, defaults)

        light_defaults = [
            {"title": "Источник света 1", "pos": (2000, 1500, -500), "I0": 800, "color": (1.0, 1.0, 1.0)},
            {"title": "Источник света 2", "pos": (-1000, -1000, -800), "I0": 300, "color": (1.0, 0.8, 0.5)}
        ]
        for idx, defaults in enumerate(light_defaults, start=1):
            self._build_light_section(params_frame, idx, defaults)

        self.last_pil = None
        self.last_projections = None

        self.render()

    # --- Фабрика числовых контролов (ярлык + поле + слайдер) ---
    def _create_labeled_entry(self, parent, label_text, default_value, row, with_slider=False, range_vals=None):
        """Строка ввода с подписью и опциональным ползунком (для числовых параметров)."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2,
                   sticky='ew', pady=5, padx=10)

        ttk.Label(frame, text=label_text, width=25,
                  anchor='w').pack(side=tk.LEFT)

        var = tk.StringVar(value=str(default_value))
        entry = ttk.Entry(frame, textvariable=var, width=12)
        entry.pack(side=tk.RIGHT, padx=(10, 0))

        if with_slider and range_vals:
            slider = ttk.Scale(frame, from_=range_vals[0], to=range_vals[1], orient=tk.HORIZONTAL,
                               command=lambda v: var.set(f"{float(v):.2f}"))
            slider.set(default_value)
            slider.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 5))

        return var

    # --- Создание строки выбора цвета (RGB и кнопка вызова палитры) ---
    def _create_color_picker(self, parent, label_text, default_rgb, row):
        """Три поля/кнопка выбора RGB-цвета (значения 0..1)."""
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, columnspan=2,
                   sticky='ew', pady=5, padx=10)

        ttk.Label(frame, text=label_text, width=25,
                  anchor='w').pack(side=tk.LEFT)

        r_var = tk.StringVar(value=str(default_rgb[0]))
        g_var = tk.StringVar(value=str(default_rgb[1]))
        b_var = tk.StringVar(value=str(default_rgb[2]))

        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(float(default_rgb[0]) * 255),
            int(float(default_rgb[1]) * 255),
            int(float(default_rgb[2]) * 255)
        )

        color_btn = tk.Button(frame, text="  ", bg=color_hex, width=3, relief='raised', bd=2,
                              command=lambda: self._choose_color(r_var, g_var, b_var, color_btn))
        color_btn.pack(side=tk.RIGHT, padx=(5, 0))

        return r_var, g_var, b_var

    # --- Обработчик выбора цвета через системный диалог ---
    def _choose_color(self, r_var, g_var, b_var, button):
        """Открывает системный диалог выбора цвета и обновляет значения RGB."""
        color = colorchooser.askcolor(title="Выберите цвет")
        if color[1]:
            button.config(bg=color[1])
            rgb = tuple(c / 255.0 for c in color[0])
            r_var.set(f"{rgb[0]:.3f}")
            g_var.set(f"{rgb[1]:.3f}")
            b_var.set(f"{rgb[2]:.3f}")

    # --- Секция настроек экрана и наблюдателя ---
    def _build_camera_section(self, parent):
        """Параметры экрана, разрешения и позиции наблюдателя."""
        frame = ttk.LabelFrame(parent, text="Экран и наблюдатель", padding=15)
        frame.pack(fill=tk.X, padx=10, pady=10)

        self.W_var = self._create_labeled_entry(frame, "Ширина экрана W (мм)", 800, 0)
        self.H_var = self._create_labeled_entry(frame, "Высота экрана H (мм)", 600, 1)
        self.Wres_var = self._create_labeled_entry(frame, "Разрешение Wres (px)", 400, 2)
        self.Hres_var = self._create_labeled_entry(frame, "Разрешение Hres (px)", 300, 3)
        self.zscr_var = self._create_labeled_entry(frame, "Положение экрана z_scr (мм)", 0, 4)
        self.zO_var = self._create_labeled_entry(frame, "Положение наблюдателя zO (мм)", -5000, 5)

    # --- Секция управления параметрами конкретной сферы ---
    def _build_sphere_section(self, parent, idx, defaults):
        """Создаёт группу контролов для одной сферы (позиция, материал, цвет)."""
        frame = ttk.LabelFrame(parent, text=defaults.get("title", f"Сфера {idx}"), padding=15)
        frame.pack(fill=tk.X, padx=10, pady=10)

        geom_frame = ttk.LabelFrame(frame, text="Геометрия", padding=10)
        geom_frame.pack(fill=tk.X, pady=(0, 8))
        setattr(self, f"R{idx}_var", self._create_labeled_entry(
            geom_frame, f"Радиус R{idx} (мм)", defaults["radius"], 0, True, (50, 500)))
        setattr(self, f"C{idx}x_var", self._create_labeled_entry(
            geom_frame, f"Центр X{idx} (мм)", defaults["center"][0], 1))
        setattr(self, f"C{idx}y_var", self._create_labeled_entry(
            geom_frame, f"Центр Y{idx} (мм)", defaults["center"][1], 2))
        setattr(self, f"C{idx}z_var", self._create_labeled_entry(
            geom_frame, f"Центр Z{idx} (мм)", defaults["center"][2], 3))

        mat_frame = ttk.LabelFrame(frame, text="Материал", padding=10)
        mat_frame.pack(fill=tk.X, pady=(0, 8))
        setattr(self, f"kd{idx}_var", self._create_labeled_entry(
            mat_frame, f"Диффузия kd{idx}", defaults["kd"], 0, True, (0, 1)))
        setattr(self, f"ks{idx}_var", self._create_labeled_entry(
            mat_frame, f"Зеркальность ks{idx}", defaults["ks"], 1, True, (0, 1)))
        setattr(self, f"sh{idx}_var", self._create_labeled_entry(
            mat_frame, f"Блеск shininess{idx}", defaults["shininess"], 2, True, (1, 200)))

        color_frame = ttk.LabelFrame(frame, text="Цвет", padding=10)
        color_frame.pack(fill=tk.X)
        r_var, g_var, b_var = self._create_color_picker(
            color_frame, f"Цвет сферы {idx}", defaults["color"], 0)
        setattr(self, f"C{idx}r_var", r_var)
        setattr(self, f"C{idx}g_var", g_var)
        setattr(self, f"C{idx}b_var", b_var)

    # --- Секция параметров точечного источника света ---
    def _build_light_section(self, parent, idx, defaults):
        """Параметры точечного источника света (координаты, мощность, цвет)."""
        frame = ttk.LabelFrame(parent, text=defaults.get("title", f"Источник света {idx}"), padding=15)
        frame.pack(fill=tk.X, padx=10, pady=10)

        setattr(self, f"L{idx}x_var", self._create_labeled_entry(
            frame, f"Позиция X{idx} (мм)", defaults["pos"][0], 0))
        setattr(self, f"L{idx}y_var", self._create_labeled_entry(
            frame, f"Позиция Y{idx} (мм)", defaults["pos"][1], 1))
        setattr(self, f"L{idx}z_var", self._create_labeled_entry(
            frame, f"Позиция Z{idx} (мм)", defaults["pos"][2], 2))
        setattr(self, f"I{idx}_var", self._create_labeled_entry(
            frame, f"Интенсивность I0{idx} (Вт/ср)", defaults["I0"], 3, True, (0, 2000)))
        r_var, g_var, b_var = self._create_color_picker(
            frame, f"Цвет света {idx}", defaults["color"], 4)
        setattr(self, f"LC{idx}r_var", r_var)
        setattr(self, f"LC{idx}g_var", g_var)
        setattr(self, f"LC{idx}b_var", b_var)

    # --- Безопасное преобразование Tk-переменной в float ---
    def _safe_float(self, var):
        """Пытается считать float из Tk-переменной, иначе возвращает None."""
        try:
            return float(var.get())
        except (ValueError, tk.TclError):
            return None

    # --- Обработчик изменения размеров экрана ---
    def _on_WH_change(self, *_):
        """Поддерживает квадратные пиксели при изменении размеров экрана."""
        if self._sync_lock:
            return
        try:
            self._sync_lock = True
            W = self._safe_float(self.W_var)
            H = self._safe_float(self.H_var)
            Wres = self._safe_float(self.Wres_var)
            if not all(v is not None for v in (W, H, Wres)):
                return
            if W <= 0 or H <= 0 or Wres <= 0:
                return
            pixel = W / Wres
            Hres = max(1, int(round(H / pixel)))
            self.Hres_var.set(str(Hres))
        finally:
            self._sync_lock = False

    # --- Обработчик изменения горизонтального разрешения ---
    def _on_Wres_change(self, *_):
        """Пересчитывает Hres при изменении Wres (для квадратных пикселей)."""
        if self._sync_lock:
            return
        try:
            self._sync_lock = True
            W = self._safe_float(self.W_var)
            H = self._safe_float(self.H_var)
            Wres = self._safe_float(self.Wres_var)
            if not all(v is not None for v in (W, H, Wres)):
                return
            if W <= 0 or H <= 0 or Wres <= 0:
                return
            pixel = W / Wres
            Hres = max(1, int(round(H / pixel)))
            self.Hres_var.set(str(Hres))
        finally:
            self._sync_lock = False

    # --- Обработчик изменения вертикального разрешения ---
    def _on_Hres_change(self, *_):
        """Пересчитывает Wres при изменении Hres (для квадратных пикселей)."""
        if self._sync_lock:
            return
        try:
            self._sync_lock = True
            W = self._safe_float(self.W_var)
            H = self._safe_float(self.H_var)
            Hres = self._safe_float(self.Hres_var)
            if not all(v is not None for v in (W, H, Hres)):
                return
            if W <= 0 or H <= 0 or Hres <= 0:
                return
            pixel = H / Hres
            Wres = max(1, int(round(W / pixel)))
            self.Wres_var.set(str(Wres))
        finally:
            self._sync_lock = False

    # --- Обновление мини-превью ортогональных проекций ---
    def _update_projection_views(self):
        """Обновляет мини-превью трёх проекций в левой колонке."""
        if not self.last_projections:
            for lbl in self.projection_labels.values():
                lbl.config(text='Нет данных', image='')
            return

        max_w, max_h = 240, 240
        for key, label in self.projection_labels.items():
            proj = self.last_projections.get(key)
            if not proj:
                label.config(text='Нет данных', image='')
                continue
            img = Image.fromarray(proj['image'])
            scale = min(max_w / img.width, max_h / img.height, 1.0)
            new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            if new_size != img.size:
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(img)
            label.config(image=tk_img, text='')
            self.projection_images[key] = tk_img

    # --- Основной обработчик кнопки Render ---
    def render(self):
        """Собирает параметры сцены, рендерит изображение и обновляет превью."""
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres = int(self.Hres_var.get())
            z_scr = float(self.zscr_var.get())
            zO = float(self.zO_var.get())

            # Собираем параметры для двух сфер: центр, радиус, материал и цвет
            spheres = [
                {
                    'center': (float(self.C1x_var.get()), float(self.C1y_var.get()), float(self.C1z_var.get())),
                    'radius': float(self.R1_var.get()),
                    'kd': float(self.kd1_var.get()),
                    'ks': float(self.ks1_var.get()),
                    'shininess': float(self.sh1_var.get()),
                    'color': (float(self.C1r_var.get()), float(self.C1g_var.get()), float(self.C1b_var.get()))
                },
                {
                    'center': (float(self.C2x_var.get()), float(self.C2y_var.get()), float(self.C2z_var.get())),
                    'radius': float(self.R2_var.get()),
                    'kd': float(self.kd2_var.get()),
                    'ks': float(self.ks2_var.get()),
                    'shininess': float(self.sh2_var.get()),
                    'color': (float(self.C2r_var.get()), float(self.C2g_var.get()), float(self.C2b_var.get()))
                }
            ]

            # Аналогично формируем список источников света
            lights = [
                {
                    'pos': (float(self.L1x_var.get()), float(self.L1y_var.get()), float(self.L1z_var.get())),
                    'I0': float(self.I1_var.get()),
                    'color': (float(self.LC1r_var.get()), float(self.LC1g_var.get()), float(self.LC1b_var.get()))
                },
                {
                    'pos': (float(self.L2x_var.get()), float(self.L2y_var.get()), float(self.L2z_var.get())),
                    'I0': float(self.I2_var.get()),
                    'color': (float(self.LC2r_var.get()), float(self.LC2g_var.get()), float(self.LC2b_var.get()))
                }
            ]

            img_uint8, img_float, I_max, I_min = render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights)
            pil_img = Image.fromarray(img_uint8, mode="RGB")
            self.last_pil = pil_img

            display_size = (min(700, Wres), min(700, Hres))
            display_img = pil_img.resize(display_size, Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(display_img)
            self.image_label.config(image=self.tk_img)
            self.info_var.set(f"Яркость: Max = {I_max:.3g}, Min>0 = {I_min:.3g}")

            self.last_projections = render_projections(W, H, Wres, Hres, spheres, lights)
            self._update_projection_views()

        except Exception as e:
            messagebox.showerror("Ошибка рендеринга", str(e))

    # --- Сохранение главного изображения ---
    def save(self):
        """Сохраняет последне отрендеренное изображение в PNG."""
        if self.last_pil is None:
            messagebox.showwarning("Нет изображения", "Сначала нажмите «РЕНДЕРИТЬ».")
            return
        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if filename:
            self.last_pil.save(filename)

    # --- Формирование коллажа из трёх проекций ---
    def _create_composite_image(self, projections, wres, hres):
        """
        Создает сетку 2x2:
        [ Пусто ]    [ Вид XZ (Y) ]
        [ Вид XY (Z) ] [ Вид YZ (X) ]
        """
        margin = 10
        text_height = 30

        # Размеры холста
        total_width = wres * 2 + margin * 3
        total_height = (hres + text_height) * 2 + margin * 3

        # Фон темный, как в UI
        composite = Image.new('RGB', (total_width, total_height), color='#2b2b2b')
        draw = ImageDraw.Draw(composite)

        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()

        text_color = '#ffffff'

        # Вспомогательная функция вставки
        def paste_view(key, col, row):
            proj = projections[key]
            img = Image.fromarray(proj['image'])

            x = margin + col * (wres + margin)
            y = margin + row * (hres + text_height + margin)

            # Рисуем заголовок
            title = proj['name']
            bbox = draw.textbbox((0, 0), title, font=font)
            title_w = bbox[2] - bbox[0]
            title_x = x + (wres - title_w) // 2
            draw.text((title_x, y), title, fill=text_color, font=font)

            # Вставляем картинку
            composite.paste(img, (x, y + text_height))

        # 1. Правый верхний - Вид XZ (Y)
        paste_view('horizontal', 1, 0)

        # 2. Левый нижний - Вид XY (Z)
        paste_view('frontal', 0, 1)

        # 3. Правый нижний - Вид YZ (X)
        paste_view('profile', 1, 1)

        return composite

    # --- Сохранение коллажа проекций ---
    def save_projections(self):
        """Сохраняет коллаж из трёх ортогональных проекций."""
        if self.last_projections is None:
            messagebox.showwarning("Нет проекций", "Сначала выполните рендер сцены.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if not filename:
            return

        try:
            wres = int(self.Wres_var.get())
            hres = int(self.Hres_var.get())

            # Генерируем финальную картинку в полном разрешении
            composite = self._create_composite_image(self.last_projections, wres, hres)
            composite.save(filename)
            messagebox.showinfo("Успешно", f"Сохранено: {filename}")

        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))


if __name__ == "__main__":
    app = ModernSceneApp()
    app.mainloop()
