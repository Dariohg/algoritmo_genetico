import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from genetico.algoritmo import AlgoritmoGenetico
from funciones.objetivo import funcion_objetivo
from visualizacion.graficador import graficar_evolucion, graficar_funcion


class AplicacionAG(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Algoritmo Genético - Maximización de f(x)")
        self.geometry("1200x800")
        self.configure(background='white')

        # Crear algoritmo genético con valores predeterminados
        self.algoritmo = AlgoritmoGenetico(
            funcion_objetivo=funcion_objetivo,
            rango_min=10.60,
            rango_max=18.20,
            precision=0.04,
            tamano_poblacion=100,
            tasa_mutacion_individuo=0.3,
            tasa_mutacion_gen=0.1,
            max_generaciones=100,
            factor_crecimiento=1.5
        )

        # Crear interfaz
        self.crear_interfaz()

        # Configuración inicial de gráficos
        self.actualizar_graficos()

    def crear_interfaz(self):
        """Crea la interfaz gráfica de la aplicación."""
        # Frame principal
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Panel izquierdo (controles)
        left_frame = ttk.LabelFrame(main_frame, text="Configuración")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)

        # Panel derecho (gráficos)
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame para la gráfica de evolución
        self.evolucion_frame = ttk.LabelFrame(right_frame, text="Evolución del Fitness")
        self.evolucion_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Frame para la gráfica de la función
        self.funcion_frame = ttk.LabelFrame(right_frame, text="Función Objetivo")
        self.funcion_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Añadir controles
        self.crear_controles(left_frame)

    def crear_controles(self, parent):
        """Crea los controles de la aplicación."""
        # Parámetros del algoritmo
        params_frame = ttk.LabelFrame(parent, text="Parámetros del Algoritmo")
        params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Tamaño de población
        ttk.Label(params_frame, text="Tamaño de población:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.tamano_poblacion_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.tamano_poblacion_var, width=10).grid(row=0, column=1, padx=5, pady=2)

        # Tasa de mutación individuo (PMI)
        ttk.Label(params_frame, text="Tasa mutación individuo (PMI):").grid(row=1, column=0, sticky=tk.W, padx=5,
                                                                            pady=2)
        self.pmi_var = tk.StringVar(value="0.3")
        ttk.Entry(params_frame, textvariable=self.pmi_var, width=10).grid(row=1, column=1, padx=5, pady=2)

        # Tasa de mutación gen (PMG)
        ttk.Label(params_frame, text="Tasa mutación gen (PMG):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.pmg_var = tk.StringVar(value="0.1")
        ttk.Entry(params_frame, textvariable=self.pmg_var, width=10).grid(row=2, column=1, padx=5, pady=2)

        # Factor de crecimiento
        ttk.Label(params_frame, text="Factor de crecimiento:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.factor_crecimiento_var = tk.StringVar(value="1.5")
        ttk.Entry(params_frame, textvariable=self.factor_crecimiento_var, width=10).grid(row=3, column=1, padx=5,
                                                                                         pady=2)

        # Máximo de generaciones
        ttk.Label(params_frame, text="Máximo de generaciones:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_generaciones_var = tk.StringVar(value="100")
        ttk.Entry(params_frame, textvariable=self.max_generaciones_var, width=10).grid(row=4, column=1, padx=5, pady=2)

        # Botones
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(buttons_frame, text="Inicializar", command=self.inicializar).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Paso", command=self.paso).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Evolucionar", command=self.evolucionar).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(buttons_frame, text="Mostrar Resultado", command=self.mostrar_resultado).pack(fill=tk.X, padx=5,
                                                                                                 pady=2)

        # Separador
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)

        # Información de ejecución
        info_frame = ttk.LabelFrame(parent, text="Información")
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # Generación actual
        ttk.Label(info_frame, text="Generación actual:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.generacion_label = ttk.Label(info_frame, text="0")
        self.generacion_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        # Mejor fitness
        ttk.Label(info_frame, text="Mejor fitness:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.mejor_fitness_label = ttk.Label(info_frame, text="-")
        self.mejor_fitness_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        # Mejor solución
        ttk.Label(info_frame, text="Mejor solución (x):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.mejor_solucion_label = ttk.Label(info_frame, text="-")
        self.mejor_solucion_label.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        # Log de ejecución
        log_frame = ttk.LabelFrame(parent, text="Log de Ejecución")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = tk.Text(log_frame, height=10, width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Scrollbar para el log
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def log(self, mensaje):
        """Añade un mensaje al log de ejecución."""
        self.log_text.insert(tk.END, mensaje + "\n")
        self.log_text.see(tk.END)

    def actualizar_graficos(self):
        """Actualiza los gráficos de la aplicación."""
        # Limpiar frames
        for widget in self.evolucion_frame.winfo_children():
            widget.destroy()

        for widget in self.funcion_frame.winfo_children():
            widget.destroy()

        # Obtener estadísticas
        stats = self.algoritmo.obtener_estadisticas()

        # Gráfico de evolución
        if stats['generacion_actual'] > 0:
            fig_evolucion = graficar_evolucion(
                stats['mejor_fitness_historico'],
                stats['fitness_promedio_historico']
            )

            canvas_evolucion = FigureCanvasTkAgg(fig_evolucion, self.evolucion_frame)
            canvas_evolucion.draw()
            canvas_evolucion.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Gráfico de función
        fig_funcion = graficar_funcion(
            funcion_objetivo,
            self.algoritmo.rango_min,
            self.algoritmo.rango_max,
            stats['mejor_valor_real']
        )

        canvas_funcion = FigureCanvasTkAgg(fig_funcion, self.funcion_frame)
        canvas_funcion.draw()
        canvas_funcion.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Actualizar etiquetas de información
        self.generacion_label.config(text=str(stats['generacion_actual']))

        if stats['mejor_fitness'] != -np.inf:
            self.mejor_fitness_label.config(text=f"{stats['mejor_fitness']:.6f}")
        else:
            self.mejor_fitness_label.config(text="-")

        if stats['mejor_valor_real'] is not None:
            self.mejor_solucion_label.config(text=f"{stats['mejor_valor_real']:.6f}")
        else:
            self.mejor_solucion_label.config(text="-")

    def inicializar(self):
        """Inicializa el algoritmo genético con los parámetros actuales."""
        try:
            # Obtener parámetros de los controles
            tamano_poblacion = int(self.tamano_poblacion_var.get())
            pmi = float(self.pmi_var.get())
            pmg = float(self.pmg_var.get())
            factor_crecimiento = float(self.factor_crecimiento_var.get())
            max_generaciones = int(self.max_generaciones_var.get())

            # Validar parámetros
            if tamano_poblacion <= 0 or pmi < 0 or pmi > 1 or pmg < 0 or pmg > 1 or factor_crecimiento <= 0:
                raise ValueError("Parámetros inválidos")

            # Crear nuevo algoritmo
            self.algoritmo = AlgoritmoGenetico(
                funcion_objetivo=funcion_objetivo,
                rango_min=10.60,
                rango_max=18.20,
                precision=0.04,
                tamano_poblacion=tamano_poblacion,
                tasa_mutacion_individuo=pmi,
                tasa_mutacion_gen=pmg,
                max_generaciones=max_generaciones,
                factor_crecimiento=factor_crecimiento
            )

            self.log(f"Algoritmo inicializado con: población={tamano_poblacion}, PMI={pmi}, PMG={pmg}")
            self.actualizar_graficos()

        except ValueError as e:
            messagebox.showerror("Error", f"Parámetros inválidos: {str(e)}")

    def paso(self):
        """Ejecuta un paso de evolución."""
        try:
            mejor_fitness, fitness_promedio, _ = self.algoritmo.paso_generacion()

            gen_actual = self.algoritmo.generacion_actual
            self.log(f"Generación {gen_actual}: Mejor={mejor_fitness:.6f}, Promedio={fitness_promedio:.6f}")

            self.actualizar_graficos()

        except Exception as e:
            messagebox.showerror("Error", f"Error al ejecutar paso: {str(e)}")

    def evolucionar(self):
        """Evoluciona el algoritmo hasta el máximo de generaciones."""
        try:
            # Generaciones restantes
            gen_actual = self.algoritmo.generacion_actual
            gen_max = int(self.max_generaciones_var.get())
            pasos_restantes = gen_max - gen_actual

            if pasos_restantes <= 0:
                messagebox.showinfo("Información", "Ya se alcanzó el máximo de generaciones")
                return

            # Ejecutar evolución
            _, mejor_fitness, mejor_valor = self.algoritmo.evolucionar(pasos_restantes)

            self.log(f"Evolución completada: {pasos_restantes} generaciones")
            self.log(f"Mejor fitness: {mejor_fitness:.6f}")
            self.log(f"Mejor solución: x = {mejor_valor:.6f}")

            self.actualizar_graficos()

        except Exception as e:
            messagebox.showerror("Error", f"Error al evolucionar: {str(e)}")

    def mostrar_resultado(self):
        """Muestra los resultados del algoritmo."""
        stats = self.algoritmo.obtener_estadisticas()

        if stats['mejor_valor_real'] is None:
            messagebox.showinfo("Información", "No hay resultados disponibles")
            return

        mensaje = (
            f"Resultados del Algoritmo Genético\n"
            f"==============================\n\n"
            f"Problema: Maximizar f(x) = ln(10 + 3 cos(7x) - 5 sen(13x) + abs(x))\n"
            f"Rango: [{self.algoritmo.rango_min}, {self.algoritmo.rango_max}]\n"
            f"Precisión: {self.algoritmo.precision}\n\n"
            f"Mejor solución encontrada:\n"
            f"x = {stats['mejor_valor_real']:.6f}\n"
            f"f(x) = {stats['mejor_fitness']:.6f}\n\n"
            f"Generaciones ejecutadas: {stats['generacion_actual']}\n"
            f"Tamaño de población: {self.algoritmo.tamano_poblacion}\n"
            f"Tasa de mutación individuo (PMI): {self.algoritmo.tasa_mutacion_individuo}\n"
            f"Tasa de mutación gen (PMG): {self.algoritmo.tasa_mutacion_gen}\n"
        )

        messagebox.showinfo("Resultados", mensaje)


if __name__ == "__main__":
    app = AplicacionAG()
    app.mainloop()