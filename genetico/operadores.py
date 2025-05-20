import numpy as np
from typing import List, Tuple


def emparejamiento_aleatorio(poblacion: np.ndarray) -> List[Tuple[int, int]]:
    """
    Cada individuo genera una pareja con otro individuo aleatorio, incluyéndose a sí mismo.

    Args:
        poblacion: Población binaria

    Returns:
        Lista de parejas (tuplas de índices)
    """
    tam_poblacion = len(poblacion)
    parejas = []

    for i in range(tam_poblacion):
        # Seleccionar otro individuo aleatorio (puede ser el mismo)
        j = np.random.randint(0, tam_poblacion)
        parejas.append((i, j))

    return parejas


def cruza_dos_puntos(padre1: np.ndarray, padre2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realiza cruza con dos puntos de corte en posiciones aleatorias.

    Args:
        padre1: Primer individuo
        padre2: Segundo individuo

    Returns:
        Dos individuos hijos
    """
    longitud = len(padre1)

    # Asegurarse de que el tamaño sea suficiente para dos puntos de cruza
    if longitud <= 2:
        return padre1.copy(), padre2.copy()

    # Generar dos puntos de cruza aleatorios distintos
    puntos_cruza = sorted(np.random.choice(range(1, longitud), size=2, replace=False))

    # Crear los hijos
    hijo1 = np.concatenate([
        padre1[:puntos_cruza[0]],
        padre2[puntos_cruza[0]:puntos_cruza[1]],
        padre1[puntos_cruza[1]:]
    ])

    hijo2 = np.concatenate([
        padre2[:puntos_cruza[0]],
        padre1[puntos_cruza[0]:puntos_cruza[1]],
        padre2[puntos_cruza[1]:]
    ])

    return hijo1, hijo2


def mutacion_complemento(
        poblacion: np.ndarray,
        pmi: float,
        pmg: float
) -> np.ndarray:
    """
    Aplica mutación siguiendo los criterios PMI y PMG.

    Args:
        poblacion: Población binaria
        pmi: Porcentaje de mutación del individuo
        pmg: Porcentaje de mutación del gen

    Returns:
        Población mutada
    """
    poblacion_mutada = poblacion.copy()
    tam_poblacion, longitud_individuo = poblacion.shape

    for i in range(tam_poblacion):
        # Decidir si el individuo muta
        if np.random.random() > pmi:  # Solo mutan los que NO superan el umbral
            # Para cada gen del individuo
            for j in range(longitud_individuo):
                # Decidir si el gen muta
                if np.random.random() > pmg:  # Solo mutan los genes que NO superan el umbral
                    # Complementar el valor del gen (0->1, 1->0)
                    poblacion_mutada[i, j] = 1 - poblacion_mutada[i, j]

    return poblacion_mutada


def poda_aleatoria_conservando_mejor(
        poblacion: np.ndarray,
        fitness: np.ndarray,
        tamano_nueva_poblacion: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elimina individuos aleatoriamente, conservando al mejor.

    Args:
        poblacion: Población binaria
        fitness: Valores de fitness de la población
        tamano_nueva_poblacion: Tamaño deseado de la nueva población

    Returns:
        Nueva población y sus valores de fitness
    """
    if len(poblacion) <= tamano_nueva_poblacion:
        return poblacion, fitness

    # Encontrar el índice del mejor individuo
    idx_mejor = np.argmax(fitness)

    # Seleccionar índices aleatorios para conservar (excluyendo el mejor, que ya se conserva)
    indices_disponibles = list(range(len(poblacion)))
    indices_disponibles.remove(idx_mejor)
    indices_a_conservar = np.random.choice(
        indices_disponibles,
        size=tamano_nueva_poblacion - 1,
        replace=False
    )

    # Añadir el índice del mejor individuo
    indices_a_conservar = np.append(indices_a_conservar, idx_mejor)

    # Crear nueva población y sus valores de fitness
    nueva_poblacion = poblacion[indices_a_conservar]
    nuevo_fitness = fitness[indices_a_conservar]

    return nueva_poblacion, nuevo_fitness