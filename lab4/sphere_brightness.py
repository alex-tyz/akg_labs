import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
from PIL import Image


class SphereBrightnessApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ЛР 4. Расчет яркости на сфере")
        self.root.geometry("1300x860")

        self.setup_default_parameters()
        self.setup_ui()

    def setup_default_parameters(self):
        self.W = 5000.0
        self.H = 4000.0
        self.Wres = 500
        self.Hres = 400
        self.observer_z = 6500.0
        self.sphere_x = 0.0
        self.sphere_y = 0.0
        self.sphere_z = 2800.0
        self.radius = 1500.0
        self.kd = 0.7
        self.ks = 0.35
        self.shininess = 40.0
        self.ambient = 0.05
        self.light_defaults = [
            {"x": -1800.0, "y": 1200.0, "z": 5200.0, "I0": 1400.0},
            {"x": 1600.0, "y": -1000.0, "z": 4500.0, "I0": 1100.0},
        ]

    def setup_ui(self):
        self.value_labels = []

        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E), padx=(0, 10))
        left_frame.columnconfigure(0, weight=1)

        results_frame = ttk.LabelFrame(left_frame, text="Расчетные значения", padding="10")
        results_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        self.results_text = tk.Text(results_frame, height=14, width=46, wrap=tk.WORD, font=("Courier", 9))
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.results_text.config(state=tk.DISABLED)

        controls = ttk.LabelFrame(left_frame, text="Параметры", padding="5")
        controls.grid(row=1, column=0, sticky=(tk.W, tk.E))
        controls.columnconfigure(0, weight=1)

        notebook = ttk.Notebook(controls)
        notebook.grid(row=0, column=0, sticky=(tk.W, tk.E))

        self.setup_parameter_controls(notebook)

        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        ttk.Button(button_frame, text="Рассчитать", command=self.calculate).grid(row=0, column=0, padx=5, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="Сохранить изображение", command=self.save_image).grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E))

        image_frame = ttk.LabelFrame(main_frame, text="Распределение яркости", padding="5")
        image_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        image_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)

        self.image_fig = plt.Figure(figsize=(6.4, 5.4))
        self.image_canvas = FigureCanvasTkAgg(self.image_fig, master=image_frame)
        self.image_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

        self.setup_var_traces()
        self.calculate()

    def setup_parameter_controls(self, notebook):
        # Экран
        screen_frame = ttk.Frame(notebook, padding=5)
        notebook.add(screen_frame, text="Экран")
        for i in range(3):
            screen_frame.columnconfigure(i, weight=1)
        self.W_var = tk.DoubleVar(value=self.W)
        self.create_parameter_control(screen_frame, "Ширина W (мм)", self.W_var, 0, 100.0, 10000.0)
        self.H_var = tk.DoubleVar(value=self.H)
        self.create_parameter_control(screen_frame, "Высота H (мм)", self.H_var, 1, 100.0, 10000.0)
        self.Wres_var = tk.IntVar(value=self.Wres)
        self.create_parameter_control(screen_frame, "Разрешение Wres (пикс)", self.Wres_var, 2, 200, 800, integer=True)
        self.Hres_var = tk.IntVar(value=self.Hres)
        self.create_parameter_control(screen_frame, "Разрешение Hres (пикс)", self.Hres_var, 3, 200, 800, integer=True)

        # Сфера и наблюдатель
        sphere_frame = ttk.Frame(notebook, padding=5)
        notebook.add(sphere_frame, text="Сфера/Наблюдатель")
        for i in range(3):
            sphere_frame.columnconfigure(i, weight=1)
        self.sphere_x_var = tk.DoubleVar(value=self.sphere_x)
        self.create_parameter_control(sphere_frame, "xC (мм)", self.sphere_x_var, 0, -10000.0, 10000.0, format_str="{:.0f}")
        self.sphere_y_var = tk.DoubleVar(value=self.sphere_y)
        self.create_parameter_control(sphere_frame, "yC (мм)", self.sphere_y_var, 1, -10000.0, 10000.0, format_str="{:.0f}")
        self.sphere_z_var = tk.DoubleVar(value=self.sphere_z)
        self.create_parameter_control(sphere_frame, "zC (мм)", self.sphere_z_var, 2, 100.0, 10000.0)
        self.radius_var = tk.DoubleVar(value=self.radius)
        self.create_parameter_control(sphere_frame, "Радиус R (мм)", self.radius_var, 3, 100.0, 5000.0)
        self.observer_z_var = tk.DoubleVar(value=self.observer_z)
        self.create_parameter_control(sphere_frame, "zO наблюдателя (мм)", self.observer_z_var, 4, 500.0, 15000.0)

        # Материал
        material_frame = ttk.Frame(notebook, padding=5)
        notebook.add(material_frame, text="Поверхность")
        for i in range(3):
            material_frame.columnconfigure(i, weight=1)
        self.kd_var = tk.DoubleVar(value=self.kd)
        self.create_parameter_control(material_frame, "Kd (диффузный)", self.kd_var, 0, 0.0, 1.0, format_str="{:.2f}")
        self.ks_var = tk.DoubleVar(value=self.ks)
        self.create_parameter_control(material_frame, "Ks (зеркальный)", self.ks_var, 1, 0.0, 1.0, format_str="{:.2f}")
        self.shininess_var = tk.DoubleVar(value=self.shininess)
        self.create_parameter_control(material_frame, "Шероховатость n", self.shininess_var, 2, 1.0, 200.0)
        self.ambient_var = tk.DoubleVar(value=self.ambient)
        self.create_parameter_control(material_frame, "Ka (окружающий)", self.ambient_var, 3, 0.0, 0.5, format_str="{:.2f}")

        # Источники света
        lights_frame = ttk.Frame(notebook, padding=5)
        notebook.add(lights_frame, text="Источники")
        lights_frame.columnconfigure(0, weight=1)
        self.light_vars = []
        for idx, defaults in enumerate(self.light_defaults):
            light_frame = ttk.LabelFrame(lights_frame, text=f"Источник {idx + 1}", padding="5")
            light_frame.grid(row=idx, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
            vars_dict = {
                "x": tk.DoubleVar(value=defaults["x"]),
                "y": tk.DoubleVar(value=defaults["y"]),
                "z": tk.DoubleVar(value=defaults["z"]),
                "I0": tk.DoubleVar(value=defaults["I0"]),
            }
            self.create_parameter_control(light_frame, "xL (мм)", vars_dict["x"], 0, -10000.0, 10000.0, format_str="{:.0f}")
            self.create_parameter_control(light_frame, "yL (мм)", vars_dict["y"], 1, -10000.0, 10000.0, format_str="{:.0f}")
            self.create_parameter_control(light_frame, "zL (мм)", vars_dict["z"], 2, 100.0, 10000.0)
            self.create_parameter_control(light_frame, "I0 (Вт/ср)", vars_dict["I0"], 3, 0.01, 10000.0, format_str="{:.1f}")
            self.light_vars.append(vars_dict)

    def create_parameter_control(self, parent, text, var, row, min_val, max_val, suffix="", format_str="{:.0f}", integer=False):
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky=tk.W, pady=2)
        scale = ttk.Scale(parent, from_=min_val, to=max_val, variable=var, orient=tk.HORIZONTAL, length=180)
        scale.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        display = ttk.Label(parent, text=format_str.format(var.get()) + suffix, cursor="hand2")
        display.grid(row=row, column=2, padx=5, pady=2)
        display.bind(
            "<Button-1>",
            lambda event, v=var, label=text, min_v=min_val, max_v=max_val, integer=integer: self.prompt_value(label, v, min_v, max_v, integer=integer),
        )
        self.value_labels.append((var, display, suffix, format_str))

    def setup_var_traces(self):
        for var, label, suffix, fmt in self.value_labels:
            var.trace_add("write", lambda *args, v=var, lbl=label, suf=suffix, f=fmt: self.update_label(v, lbl, suf, f))

    def update_label(self, var, label_widget, suffix="", format_str="{:.0f}"):
        try:
            value = var.get()
        except tk.TclError:
            return
        label_widget.config(text=format_str.format(value) + suffix)

    def prompt_value(self, title, variable, min_val, max_val, integer=False):
        current_value = variable.get()
        prompt = f"{title}\nДиапазон: [{min_val}, {max_val}]"
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

    def calculate(self):
        brightness, mask = self.calculate_brightness_map()
        brightness_max = np.max(brightness[mask]) if np.any(mask) else 0.0

        self.brightness_data = brightness
        self.visible_mask = mask

        if brightness_max > 0:
            normalized = (brightness / brightness_max * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(brightness, dtype=np.uint8)

        self.image_data = normalized

        self.update_results_display(brightness, mask, brightness_max)
        self.render_image(normalized)

    def calculate_brightness_map(self):
        W = self.W_var.get()
        H = self.H_var.get()
        Wres = self.Wres_var.get()
        Hres = self.Hres_var.get()
        xC = self.sphere_x_var.get()
        yC = self.sphere_y_var.get()
        zC = self.sphere_z_var.get()
        R = self.radius_var.get()
        zO = self.observer_z_var.get()
        kd = self.kd_var.get()
        ks = self.ks_var.get()
        shininess = self.shininess_var.get()
        ambient = self.ambient_var.get()

        pixel_size_x = W / Wres
        pixel_size_y = H / Hres
        if abs(pixel_size_x - pixel_size_y) > 0.01:
            pixel_size = max(pixel_size_x, pixel_size_y)
            Wres = int(W / pixel_size)
            Hres = int(H / pixel_size)
            self.Wres_var.set(Wres)
            self.Hres_var.set(Hres)

        x = np.linspace(-W / 2, W / 2, Wres)
        y = np.linspace(-H / 2, H / 2, Hres)
        X, Y = np.meshgrid(x, y)

        observer = np.array([0.0, 0.0, zO])
        center = np.array([xC, yC, zC])

        dir_x = X.copy()
        dir_y = Y.copy()
        dir_z = np.full_like(X, -zO)
        dir_norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        dir_x /= dir_norm
        dir_y /= dir_norm
        dir_z /= dir_norm

        ocx = observer[0] - center[0]
        ocy = observer[1] - center[1]
        ocz = observer[2] - center[2]

        b = 2 * (ocx * dir_x + ocy * dir_y + ocz * dir_z)
        c = ocx**2 + ocy**2 + ocz**2 - R**2

        discriminant = b**2 - 4 * c
        mask = discriminant >= 0
        t = np.full_like(X, np.nan, dtype=np.float64)
        sqrt_disc = np.zeros_like(X)
        sqrt_disc[mask] = np.sqrt(np.clip(discriminant[mask], 0, None))
        t1 = np.full_like(X, np.nan, dtype=np.float64)
        t2 = np.full_like(X, np.nan, dtype=np.float64)
        t1[mask] = (-b[mask] - sqrt_disc[mask]) / 2.0
        t2[mask] = (-b[mask] + sqrt_disc[mask]) / 2.0
        t_candidate = np.where((t1 > 0) & np.isfinite(t1), t1, t2)
        mask &= t_candidate > 0
        t[mask] = t_candidate[mask]

        point_x = np.zeros_like(X)
        point_y = np.zeros_like(X)
        point_z = np.zeros_like(X)
        point_x[mask] = observer[0] + dir_x[mask] * t[mask]
        point_y[mask] = observer[1] + dir_y[mask] * t[mask]
        point_z[mask] = observer[2] + dir_z[mask] * t[mask]

        normal_x = np.zeros_like(X)
        normal_y = np.zeros_like(X)
        normal_z = np.zeros_like(X)
        normal_x[mask] = (point_x[mask] - xC) / R
        normal_y[mask] = (point_y[mask] - yC) / R
        normal_z[mask] = (point_z[mask] - zC) / R

        view_x = -dir_x
        view_y = -dir_y
        view_z = -dir_z

        brightness = np.zeros_like(X, dtype=np.float64)
        brightness[mask] = ambient

        for light in self.light_vars:
            xL = light["x"].get()
            yL = light["y"].get()
            zL = light["z"].get()
            I0 = light["I0"].get()

            light_vec_x = np.zeros_like(X)
            light_vec_y = np.zeros_like(X)
            light_vec_z = np.zeros_like(X)
            light_vec_x[mask] = xL - point_x[mask]
            light_vec_y[mask] = yL - point_y[mask]
            light_vec_z[mask] = zL - point_z[mask]

            distance = np.ones_like(X)
            distance[mask] = np.sqrt(
                light_vec_x[mask] ** 2 + light_vec_y[mask] ** 2 + light_vec_z[mask] ** 2
            )
            attenuation = np.zeros_like(X)
            attenuation[mask] = I0 / np.maximum(distance[mask] ** 2, 1e-6)

            light_dir_x = np.zeros_like(X)
            light_dir_y = np.zeros_like(X)
            light_dir_z = np.zeros_like(X)
            light_dir_x[mask] = light_vec_x[mask] / distance[mask]
            light_dir_y[mask] = light_vec_y[mask] / distance[mask]
            light_dir_z[mask] = light_vec_z[mask] / distance[mask]

            cos_theta = np.zeros_like(X)
            cos_theta[mask] = np.clip(
                normal_x[mask] * light_dir_x[mask]
                + normal_y[mask] * light_dir_y[mask]
                + normal_z[mask] * light_dir_z[mask],
                0.0,
                None,
            )

            diffuse = np.zeros_like(X)
            diffuse[mask] = kd * attenuation[mask] * cos_theta[mask]

            half_x = np.zeros_like(X)
            half_y = np.zeros_like(X)
            half_z = np.zeros_like(X)
            half_x[mask] = light_dir_x[mask] + view_x[mask]
            half_y[mask] = light_dir_y[mask] + view_y[mask]
            half_z[mask] = light_dir_z[mask] + view_z[mask]

            half_norm = np.ones_like(X)
            half_norm[mask] = np.sqrt(half_x[mask] ** 2 + half_y[mask] ** 2 + half_z[mask] ** 2)
            half_x[mask] /= np.maximum(half_norm[mask], 1e-8)
            half_y[mask] /= np.maximum(half_norm[mask], 1e-8)
            half_z[mask] /= np.maximum(half_norm[mask], 1e-8)

            spec_angle = np.zeros_like(X)
            spec_angle[mask] = np.clip(
                normal_x[mask] * half_x[mask]
                + normal_y[mask] * half_y[mask]
                + normal_z[mask] * half_z[mask],
                0.0,
                None,
            )

            specular = np.zeros_like(X)
            specular[mask] = ks * attenuation[mask] * (spec_angle[mask] ** shininess)

            brightness += diffuse + specular

        return brightness, mask

    def render_image(self, normalized):
        self.image_fig.clear()
        ax = self.image_fig.add_subplot(1, 1, 1)
        if normalized.size > 0:
            extent = [
                -self.W_var.get() / 2,
                self.W_var.get() / 2,
                -self.H_var.get() / 2,
                self.H_var.get() / 2,
            ]
            im = ax.imshow(
                normalized,
                cmap="gray",
                origin="lower",
                extent=extent,
            )
            ax.set_xlabel("X (мм)")
            ax.set_ylabel("Y (мм)")
            ax.set_title("Распределение яркости на сфере")
            self.image_fig.colorbar(im, ax=ax, label="Нормированная яркость (0-255)", fraction=0.046, pad=0.04)
        self.image_fig.tight_layout()
        self.image_canvas.draw()

    def calculate_point_brightness(self, point):
        xC = self.sphere_x_var.get()
        yC = self.sphere_y_var.get()
        zC = self.sphere_z_var.get()
        R = self.radius_var.get()
        kd = self.kd_var.get()
        ks = self.ks_var.get()
        shininess = self.shininess_var.get()
        ambient = self.ambient_var.get()
        observer = np.array([0.0, 0.0, self.observer_z_var.get()])

        center = np.array([xC, yC, zC])
        normal = point - center
        normal_norm = np.linalg.norm(normal)
        if normal_norm == 0:
            return 0.0
        normal = normal / normal_norm

        view_vec = observer - point
        view_norm = np.linalg.norm(view_vec)
        if view_norm == 0:
            return 0.0
        view_dir = view_vec / view_norm
        if np.dot(normal, view_dir) <= 0:
            return 0.0

        brightness = ambient
        for light in self.light_vars:
            light_pos = np.array([light["x"].get(), light["y"].get(), light["z"].get()])
            I0 = light["I0"].get()
            light_vec = light_pos - point
            distance = np.linalg.norm(light_vec)
            if distance <= 1e-6:
                continue
            light_dir = light_vec / distance
            cos_theta = np.dot(normal, light_dir)
            if cos_theta <= 0:
                continue
            attenuation = I0 / (distance**2)
            diffuse = kd * attenuation * cos_theta

            half_vec = light_dir + view_dir
            half_norm = np.linalg.norm(half_vec)
            specular = 0.0
            if half_norm > 1e-8:
                half_vec /= half_norm
                specular = ks * attenuation * max(np.dot(normal, half_vec), 0.0) ** shininess

            brightness += diffuse + specular

        return brightness

    def update_results_display(self, brightness, mask, max_value):
        xC = self.sphere_x_var.get()
        yC = self.sphere_y_var.get()
        zC = self.sphere_z_var.get()
        R = self.radius_var.get()
        observer = np.array([0.0, 0.0, self.observer_z_var.get()])
        center = np.array([xC, yC, zC])

        front_dir = observer - center
        if np.linalg.norm(front_dir) == 0:
            front_dir = np.array([0.0, 0.0, 1.0])
        else:
            front_dir = front_dir / np.linalg.norm(front_dir)
        front_point = center + front_dir * R
        x_point = center + np.array([R, 0.0, 0.0])
        y_point = center + np.array([0.0, R, 0.0])

        points = [
            ("Точка зрения", front_point),
            ("Точка +X", x_point),
            ("Точка +Y", y_point),
        ]

        results = []
        for label, pt in points:
            brightness_value = self.calculate_point_brightness(pt)
            results.append((label, pt, brightness_value))

        valid_values = brightness[mask]
        if valid_values.size > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
        else:
            min_val = 0.0
            max_val = 0.0

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)

        text = "Яркость на сфере (Вт/м²):\n"
        text += f"{'='*50}\n\n"
        for label, pt, value in results:
            text += f"{label} ({pt[0]:.1f}, {pt[1]:.1f}, {pt[2]:.1f}):\n"
            text += f"  L = {value:.6e}\n\n"

        text += f"{'='*50}\n"
        text += "Статистика по видимой части сферы:\n"
        text += f"  L_max = {max_val:.6e}\n"
        text += f"  L_min = {min_val:.6e}\n"
        text += f"  Нормировка выполнялась по L_max = {max_value:.6e}\n"

        self.results_text.insert(1.0, text)
        self.results_text.config(state=tk.DISABLED)

    def save_image(self):
        if not hasattr(self, "image_data"):
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        )
        if not filename:
            return

        img = Image.fromarray(self.image_data, mode="L")
        img.save(filename)
        print(f"Изображение сохранено: {filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SphereBrightnessApp(root)
    root.mainloop()
