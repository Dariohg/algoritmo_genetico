import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable, Optional
from matplotlib.figure import Figure


def graficar_evolucion(
        mejor_fitness: List[float],
        fitness_promedio: List[float],
        titulo: str = "Evolución del Fitness"
) -> Figure:
    """
    Grafica la evolución del fitness a lo largo de las generaciones.

    Args:
        mejor_fitness: Lista con el mejor fitness de cada generación
        fitness_promedio: Lista con el fitness promedio de cada generación
        titulo: Título del gráfico

    Returns:
        Figura de matplotlib
    """
    fig = plt.figure(figsize=(10, 6))

    generaciones = range(len(mejor_fitness))

    plt.plot(generaciones, mejor_fitness, 'b-', label='Mejor Fitness')
    plt.plot(generaciones, fitness_promedio, 'r-', label='Fitness Promedio')

    plt.xlabel('Generación')
    plt.ylabel('Fitness (maximización)')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return fig


def graficar_funcion(
        funcion: Callable[[float], float],
        rango_min: float,
        rango_max: float,
        mejor_valor: Optional[float] = None,
        titulo: str = "Función Objetivo",
        puntos: int = 1000
) -> Figure:
    """
    Grafica la función objetivo y opcionalmente marca el mejor valor encontrado.

    Args:
        funcion: Función objetivo
        rango_min: Valor mínimo del rango
        rango_max: Valor máximo del rango
        mejor_valor: Mejor valor encontrado (opcional)
        titulo: Título del gráfico
        puntos: Número de puntos para graficar la función

    Returns:
        Figura de matplotlib
    """
    fig = plt.figure(figsize=(10, 6))

    # Crear puntos para graficar la función
    x = np.linspace(rango_min, rango_max, puntos)
    y = np.array([funcion(xi) for xi in x])

    # Graficar función
    plt.plot(x, y, 'b-', label='Función Objetivo')

    # Marcar mejor valor si se proporciona
    if mejor_valor is not None:
        mejor_y = funcion(mejor_valor)
        plt.scatter(
            mejor_valor,
            mejor_y,
            c='r',
            s=100,
            label=f'Mejor Solución (x={mejor_valor:.4f}, f(x)={mejor_y:.4f})'
        )

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(titulo)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    return fig