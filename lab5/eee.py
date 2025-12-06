
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, filedialog


# =====================================================================
# ====================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =======================
# =====================================================================

def ray_sphere_intersect(O, D, C, R):
    """
    Находит ближайшее положительное пересечение луча O + t*D со сферой (C, R).
    Возвращает: t (массив), hit_mask (boolean)
    """
    OC = O - C
    a = np.sum(D * D, axis=-1)
    b = 2.0 * np.sum(D * OC, axis=-1)
    c = np.sum(OC * OC, axis=-1) - R ** 2
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


def render_scene(W, H, Wres, Hres, zO, z_scr, spheres, lights):
    """
    Рендерит сцену с двумя сферами, тенями и цветом по модели Блинна–Фонга.
    """
    O = np.array([0.0, 0.0, zO], dtype=np.float64)

    xs = np.linspace(-W / 2, W / 2, Wres, endpoint=False) + W / (2 * Wres)
    ys = np.linspace(-H / 2, H / 2, Hres, endpoint=False) + H / (2 * Hres)
    X, Y = np.meshgrid(xs, ys)
    Px = X.ravel()
    Py = Y.ravel()
    Pz = np.full_like(Px, z_scr)
    P_screen = np.stack([Px, Py, Pz], axis=1)

    D = P_screen - O
    D_norm = np.linalg.norm(D, axis=1, keepdims=True)
    D = D / np.maximum(D_norm, 1e-8)

    depth = np.full(P_screen.shape[0], np.inf, dtype=np.float64)
    sphere_id = np.full(P_screen.shape[0], -1, dtype=int)

    for i, sph in enumerate(spheres):
        C = np.array(sph['center'], dtype=np.float64)
        R = sph['radius']
        t, hit = ray_sphere_intersect(O, D, C, R)
        closer = t < depth
        mask = hit & closer
        depth[mask] = t[mask]
        sphere_id[mask] = i

    final_color = np.zeros((P_screen.shape[0], 3), dtype=np.float64)
    valid = sphere_id >= 0

    if not np.any(valid):
        img_rgb = np.zeros((Hres, Wres, 3))
        return (np.zeros((Hres, Wres, 3), dtype=np.uint8), img_rgb, 0.0, 0.0)

    P = np.zeros_like(D)
    P[valid] = O + D[valid] * depth[valid, np.newaxis]

    V = np.zeros_like(P)
    V[valid] = O - P[valid]
    V_norm = np.linalg.norm(V[valid], axis=1, keepdims=True)
    V[valid] = V[valid] / np.maximum(V_norm, 1e-8)

    N = np.zeros_like(P)
    centers_arr = np.array([spheres[i]['center'] for i in sphere_id[valid]], dtype=np.float64)
    N[valid] = P[valid] - centers_arr
    N_norm = np.linalg.norm(N[valid], axis=1, keepdims=True)
    N[valid] /= np.maximum(N_norm, 1e-8)

    for idx in np.where(valid)[0]:
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
            L_dir = L_dir_vec / L_dist
            light_color = np.array(light['color'], dtype=np.float64)
            I0 = light['I0']

            # Проверка тени
            shadow_ray_origin = point + normal * 1e-3
            shadow = False
            for other_sph in spheres:
                C_other = np.array(other_sph['center'], dtype=np.float64)
                R_other = other_sph['radius']
                t_shadow, hit_shadow = ray_sphere_intersect(shadow_ray_origin, L_dir, C_other, R_other)
                if np.any(hit_shadow) and np.min(t_shadow[hit_shadow]) < L_dist - 1e-3:
                    shadow = True
                    break

            if shadow:
                continue

            # Диффузная компонента
            diff = max(np.dot(normal, L_dir), 0.0)
            I_diff = sph['kd'] * I0 * diff

            # Зеркальная компонента
            H = L_dir + V[idx]
            H_norm_val = np.linalg.norm(H)
            if H_norm_val < 1e-8:
                spec = 0.0
            else:
                H = H / H_norm_val
                spec = max(np.dot(normal, H), 0.0) ** sph['shininess']
            I_spec = sph['ks'] * I0 * spec

            contrib = (I_diff + I_spec)
            color_acc += contrib * light_color * np.array(sph['color'])

        final_color[idx] = color_acc

    img_rgb = final_color.reshape((Hres, Wres, 3))

    # Максимальная и минимальная ненулевая яркость
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


# =====================================================================
# ============================= GUI ЧАСТЬ ==============================
# =====================================================================

class SliderEntry:
    """Комбинированный контрол: подпись, ползунок и поле ввода значения."""

    def __init__(self, parent, label, default, min_val, max_val, resolution=1.0, entry_width=10):
        self.min_val = min_val
        self.max_val = max_val
        self.resolution = resolution
        self.frame = ttk.Frame(parent)
        self.frame.columnconfigure(1, weight=1)

        ttk.Label(self.frame, text=label, width=24, anchor="w").grid(row=0, column=0, sticky="w", padx=(0, 5))

        self.var = tk.DoubleVar(value=float(default))
        self.entry_var = tk.StringVar(value=self._format_value(default))
        self._updating = False

        self.scale = tk.Scale(
            self.frame,
            from_=min_val,
            to=max_val,
            resolution=resolution,
            orient=tk.HORIZONTAL,
            variable=self.var,
            showvalue=False,
            length=220
        )
        self.scale.grid(row=0, column=1, sticky="ew", padx=(0, 6))

        self.entry = ttk.Entry(self.frame, textvariable=self.entry_var, width=entry_width)
        self.entry.grid(row=0, column=2, sticky="e")
        self.entry.bind("<FocusOut>", self._commit_entry)
        self.entry.bind("<Return>", self._commit_entry)

        self.var.trace_add("write", self._on_var_change)
        self.set(default)

    def _format_value(self, value):
        if self.resolution >= 1:
            return f"{value:.0f}"
        elif self.resolution >= 0.1:
            return f"{value:.1f}"
        elif self.resolution >= 0.01:
            return f"{value:.2f}"
        else:
            return f"{value:.3f}"

    def _commit_entry(self, event=None):
        try:
            value = float(self.entry_var.get())
        except ValueError:
            value = self.var.get()
        self.set(value)

    def _on_var_change(self, *args):
        if self._updating:
            return
        self.entry_var.set(self._format_value(self.var.get()))

    def _clamp(self, value):
        return max(self.min_val, min(self.max_val, value))

    def _apply_resolution(self, value):
        if self.resolution > 0:
            return round(value / self.resolution) * self.resolution
        return value

    def get(self):
        return self.var.get()

    def set(self, value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = self.min_val
        value = self._clamp(self._apply_resolution(value))
        self._updating = True
        self.var.set(value)
        self.entry_var.set(self._format_value(value))
        self._updating = False

    def trace_add(self, mode, callback):
        return self.var.trace_add(mode, callback)

    def trace_remove(self, mode, identifier):
        self.var.trace_remove(mode, identifier)


class SceneApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ЛР5: Две сферы с тенями и цветом (Блинн–Фонг)")
        self.geometry("1400x950")

        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)

        # Левая панель: изображение + ползунки масштаба и скроллы
        image_panel = ttk.Frame(main_frame)
        image_panel.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 15))
        image_panel.columnconfigure(0, weight=1)
        image_panel.rowconfigure(0, weight=1)

        self.image_canvas = tk.Canvas(image_panel, background="black", highlightthickness=0, width=720, height=720)
        self.image_canvas.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(image_panel, orient="vertical", command=self.image_canvas.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(image_panel, orient="horizontal", command=self.image_canvas.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.image_canvas.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)

        zoom_frame = ttk.Frame(image_panel)
        zoom_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        zoom_frame.columnconfigure(1, weight=1)
        ttk.Label(zoom_frame, text="Масштаб:").grid(row=0, column=0, sticky="w")
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.zoom_scale = ttk.Scale(zoom_frame, variable=self.zoom_var, from_=0.3, to=2.0, command=self._on_zoom_change)
        self.zoom_scale.grid(row=0, column=1, sticky="ew", padx=6)
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.grid(row=0, column=2, sticky="e")

        # Правая панель: параметры
        params_notebook = ttk.Notebook(main_frame)
        params_notebook.grid(row=0, column=1, sticky="nsew")
        params_notebook.enable_traversal()

        screen_tab = ttk.Frame(params_notebook, padding=8)
        spheres_tab = ttk.Frame(params_notebook, padding=8)
        lights_tab = ttk.Frame(params_notebook, padding=8)
        for tab in (screen_tab, spheres_tab, lights_tab):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        params_notebook.add(screen_tab, text="Экран/камера")
        params_notebook.add(spheres_tab, text="Сферы")
        params_notebook.add(lights_tab, text="Источники")

        # === Экран и наблюдатель (вкладка 1) ===
        self.W_var = self._add_slider_param(screen_tab, "Ширина экрана W [мм]", 800, 0, 100, 10000, 50)
        self.H_var = self._add_slider_param(screen_tab, "Высота экрана H [мм]", 600, 1, 100, 10000, 50)
        self.Wres_var = self._add_slider_param(screen_tab, "Разрешение Wres [пикс]", 400, 2, 200, 800, 10)
        self.Hres_var = self._add_slider_param(screen_tab, "Разрешение Hres [пикс]", 300, 3, 200, 800, 10)
        self.zscr_var = self._add_slider_param(screen_tab, "Положение экрана z_scr [мм]", 0, 4, -2000, 2000, 50)
        self.zO_var = self._add_slider_param(screen_tab, "Положение наблюдателя zO [мм]", -1000, 5, -5000, 5000, 50)

        self.W_var.trace_add("write", self._on_W_change)
        self.H_var.trace_add("write", self._on_H_change)
        self.Wres_var.trace_add("write", self._on_Wres_change)
        self.Hres_var.trace_add("write", self._on_Hres_change)

        # === Сферы (вкладка 2) ===
        spheres_nb = ttk.Notebook(spheres_tab)
        spheres_nb.grid(row=0, column=0, sticky="nsew")

        sphere1_tab = ttk.Frame(spheres_nb, padding=8)
        sphere2_tab = ttk.Frame(spheres_nb, padding=8)
        for tab in (sphere1_tab, sphere2_tab):
            tab.columnconfigure(0, weight=1)

        spheres_nb.add(sphere1_tab, text="Сфера 1")
        spheres_nb.add(sphere2_tab, text="Сфера 2")

        self.R1_var = self._add_slider_param(sphere1_tab, "Радиус R1 [мм]", 200, 0, 50, 1000, 10)
        self.C1x_var = self._add_slider_param(sphere1_tab, "Cx1 [мм]", -150, 1, -10000, 10000, 50)
        self.C1y_var = self._add_slider_param(sphere1_tab, "Cy1 [мм]", 0, 2, -10000, 10000, 50)
        self.C1z_var = self._add_slider_param(sphere1_tab, "Cz1 [мм]", 800, 3, 100, 5000, 50)
        self.kd1_var = self._add_slider_param(sphere1_tab, "kd1 (диффузия)", 0.7, 4, 0.0, 1.0, 0.05)
        self.ks1_var = self._add_slider_param(sphere1_tab, "ks1 (зеркальность)", 0.3, 5, 0.0, 1.0, 0.05)
        self.sh1_var = self._add_slider_param(sphere1_tab, "shininess1", 30, 6, 1, 200, 1)
        self.C1r_var = self._add_slider_param(sphere1_tab, "Цвет R1", 1.0, 7, 0.0, 1.0, 0.05)
        self.C1g_var = self._add_slider_param(sphere1_tab, "Цвет G1", 0.2, 8, 0.0, 1.0, 0.05)
        self.C1b_var = self._add_slider_param(sphere1_tab, "Цвет B1", 0.2, 9, 0.0, 1.0, 0.05)

        self.R2_var = self._add_slider_param(sphere2_tab, "Радиус R2 [мм]", 180, 0, 50, 1000, 10)
        self.C2x_var = self._add_slider_param(sphere2_tab, "Cx2 [мм]", 200, 1, -10000, 10000, 50)
        self.C2y_var = self._add_slider_param(sphere2_tab, "Cy2 [мм]", 100, 2, -10000, 10000, 50)
        self.C2z_var = self._add_slider_param(sphere2_tab, "Cz2 [мм]", 900, 3, 100, 5000, 50)
        self.kd2_var = self._add_slider_param(sphere2_tab, "kd2", 0.6, 4, 0.0, 1.0, 0.05)
        self.ks2_var = self._add_slider_param(sphere2_tab, "ks2", 0.4, 5, 0.0, 1.0, 0.05)
        self.sh2_var = self._add_slider_param(sphere2_tab, "shininess2", 50, 6, 1, 200, 1)
        self.C2r_var = self._add_slider_param(sphere2_tab, "Цвет R2", 0.2, 7, 0.0, 1.0, 0.05)
        self.C2g_var = self._add_slider_param(sphere2_tab, "Цвет G2", 0.8, 8, 0.0, 1.0, 0.05)
        self.C2b_var = self._add_slider_param(sphere2_tab, "Цвет B2", 0.2, 9, 0.0, 1.0, 0.05)

        # === Источники света (вкладка 3) ===
        lights_nb = ttk.Notebook(lights_tab)
        lights_nb.grid(row=0, column=0, sticky="nsew")

        light1_tab = ttk.Frame(lights_nb, padding=8)
        light2_tab = ttk.Frame(lights_nb, padding=8)
        for tab in (light1_tab, light2_tab):
            tab.columnconfigure(0, weight=1)

        lights_nb.add(light1_tab, text="Источник 1")
        lights_nb.add(light2_tab, text="Источник 2")

        self.L1x_var = self._add_slider_param(light1_tab, "L1x [мм]", 2000, 0, -10000, 10000, 100)
        self.L1y_var = self._add_slider_param(light1_tab, "L1y [мм]", 1500, 1, -10000, 10000, 100)
        self.L1z_var = self._add_slider_param(light1_tab, "L1z [мм]", -500, 2, -5000, 5000, 50)
        self.I1_var = self._add_slider_param(light1_tab, "I01 [Вт/ср]", 800, 3, 0, 10000, 10)
        self.LC1r_var = self._add_slider_param(light1_tab, "Цвет L1 (R)", 1.0, 4, 0.0, 1.0, 0.05)
        self.LC1g_var = self._add_slider_param(light1_tab, "Цвет L1 (G)", 1.0, 5, 0.0, 1.0, 0.05)
        self.LC1b_var = self._add_slider_param(light1_tab, "Цвет L1 (B)", 1.0, 6, 0.0, 1.0, 0.05)

        self.L2x_var = self._add_slider_param(light2_tab, "L2x [мм]", -1000, 0, -10000, 10000, 100)
        self.L2y_var = self._add_slider_param(light2_tab, "L2y [мм]", -1000, 1, -10000, 10000, 100)
        self.L2z_var = self._add_slider_param(light2_tab, "L2z [мм]", -800, 2, -5000, 5000, 50)
        self.I2_var = self._add_slider_param(light2_tab, "I02 [Вт/ср]", 300, 3, 0, 10000, 10)
        self.LC2r_var = self._add_slider_param(light2_tab, "Цвет L2 (R)", 1.0, 4, 0.0, 1.0, 0.05)
        self.LC2g_var = self._add_slider_param(light2_tab, "Цвет L2 (G)", 0.8, 5, 0.0, 1.0, 0.05)
        self.LC2b_var = self._add_slider_param(light2_tab, "Цвет L2 (B)", 0.5, 6, 0.0, 1.0, 0.05)

        # === Кнопки и информация ===
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky="sw", pady=(10, 0))

        self.info_var = tk.StringVar(value="Max = 0.0, Min>0 = 0.0")
        info_label = ttk.Label(control_frame, textvariable=self.info_var, foreground="blue", font=("TkDefaultFont", 10, "bold"))
        info_label.pack(anchor="w", pady=5)

        btn_render = ttk.Button(control_frame, text="Render", command=self.render, width=20)
        btn_render.pack(pady=5)

        btn_save = ttk.Button(control_frame, text="Save image", command=self.save, width=20)
        btn_save.pack(pady=5)

        self.last_pil = None
        self.tk_img = None
        self.render()  # первый рендер

    def _add_slider_param(self, parent, label_text, default_value, row, min_val, max_val, resolution, entry_width=10):
        """Создаёт строку с подписью, ползунком и полем ввода значения."""
        slider = SliderEntry(parent, label_text, default_value, min_val, max_val, resolution, entry_width)
        slider.frame.grid(row=row, column=0, sticky="ew", pady=3)
        return slider

    def _on_W_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Hres = int(self.Hres_var.get())
            Wres_new = int(round(Hres * W / H))
            self.Wres_var.set(Wres_new)
        except:
            pass

    def _on_H_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres_new = int(round(Wres * H / W))
            self.Hres_var.set(Hres_new)
        except:
            pass

    def _on_Wres_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres_new = int(round(Wres * H / W))
            self.Hres_var.set(Hres_new)
        except:
            pass

    def _on_Hres_change(self, *args):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Hres = int(self.Hres_var.get())
            Wres_new = int(round(Hres * W / H))
            self.Wres_var.set(Wres_new)
        except:
            pass

    def render(self):
        try:
            W = float(self.W_var.get())
            H = float(self.H_var.get())
            Wres = int(self.Wres_var.get())
            Hres = int(self.Hres_var.get())  # Теперь Hres тоже можно менять!
            z_scr = float(self.zscr_var.get())
            zO = float(self.zO_var.get())

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

            self._update_preview()

            # Обновляем информацию о яркости
            self.info_var.set(f"Max = {I_max:.3g}, Min>0 = {I_min:.3g}")
            pil_img.save("lab5_result.png")

        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def save(self):
        if self.last_pil is None:
            messagebox.showwarning("Нет изображения", "Сначала нажмите «Render».")
            return
        filename = filedialog.asksaveasfilename(
            title="Сохранить изображение",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if filename:
            try:
                self.last_pil.save(filename)
                messagebox.showinfo("Сохранено", f"Изображение сохранено:\n{filename}")
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))

    def _on_zoom_change(self, *args):
        """Обновляет подпись и превью при смене масштаба."""
        if self.zoom_label is not None:
            self.zoom_label.config(text=f"{self.zoom_var.get() * 100:.0f}%")
        self._update_preview()

    def _update_preview(self):
        """Перерисовывает изображение в соответствии с ползунком масштаба и включает скроллы."""
        if self.last_pil is None:
            self.image_canvas.delete("all")
            self.image_canvas.configure(scrollregion=(0, 0, 0, 0))
            return

        zoom = max(0.3, min(self.zoom_var.get(), 2.0))
        new_w = max(1, int(self.last_pil.width * zoom))
        new_h = max(1, int(self.last_pil.height * zoom))
        if zoom != 1.0:
            display_img = self.last_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        else:
            display_img = self.last_pil
        self.tk_img = ImageTk.PhotoImage(display_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.image_canvas.configure(scrollregion=(0, 0, display_img.width, display_img.height))


# =====================================================================
# ======================== ЗАПУСК ПРИЛОЖЕНИЯ ==========================
# =====================================================================

if __name__ == "__main__":
    app = SceneApp()
    app.mainloop()
