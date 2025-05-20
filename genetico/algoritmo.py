import numpy as np
from typing import Callable, Tuple, List
import time

from genetico.operadores import (
    emparejamiento_aleatorio,
    cruza_dos_puntos,
    mutacion_complemento,
    poda_aleatoria_conservando_mejor
)
from genetico.codificacion import (
    calcular_bits_necesarios,
    inicializar_poblacion_binaria,
    binario_a_real
)


class AlgoritmoGenetico:
    def __init__(
            self,
            funcion_objetivo: Callable[[float], float],
            rango_min: float = 10.60,
            rango_max: float = 18.20,
            precision: float = 0.04,
            tamano_poblacion: int = 100,
            tasa_mutacion_individuo: float = 0.3,
            tasa_mutacion_gen: float = 0.1,
            max_generaciones: int = 50,
            factor_crecimiento: float = 1.5
    ):
        """
        Inicializa el algoritmo genético.

        Args:
            funcion_objetivo: Función a maximizar
            rango_min: Valor mínimo del rango
            rango_max: Valor máximo del rango
            precision: Precisión requerida
            tamano_poblacion: Número de individuos en la población
            tasa_mutacion_individuo: Umbral PMI (porcentaje de mutación del individuo)
            tasa_mutacion_gen: Umbral PMG (porcentaje de mutación del gen)
            max_generaciones: Número máximo de generaciones
            factor_crecimiento: Factor de crecimiento de la población tras cruza
        """
        self.funcion_objetivo = funcion_objetivo
        self.rango_min = rango_min
        self.rango_max = rango_max
        self.precision = precision
        self.tamano_poblacion = tamano_poblacion
        self.tasa_mutacion_individuo = tasa_mutacion_individuo
        self.tasa_mutacion_gen = tasa_mutacion_gen
        self.max_generaciones = max_generaciones
        self.factor_crecimiento = factor_crecimiento

        # Calcular bits necesarios
        self.bits = calcular_bits_necesarios(rango_min, rango_max, precision)

        # Crear población inicial
        self.poblacion = inicializar_poblacion_binaria(tamano_poblacion, self.bits)

        # Historial para graficar
        self.mejor_fitness_historico = []
        self.fitness_promedio_historico = []
        self.mejor_individuo_historico = []
        self.generacion_actual = 0

        # Para almacenar resultados
        self.mejor_solucion = None
        self.mejor_fitness = -np.inf

    def _evaluar_poblacion(self) -> np.ndarray:
        """
        Evalúa el fitness de todos los individuos en la población.

        Returns:
            Array con los valores de fitness
        """
        fitness = np.zeros(len(self.poblacion))

        for i, individuo in enumerate(self.poblacion):
            # Convertir de binario a valor real
            valor_real = binario_a_real(
                individuo,
                self.rango_min,
                self.rango_max,
                self.bits
            )

            # Evaluar la función objetivo
            fitness[i] = self.funcion_objetivo(valor_real)

        return fitness

    def _cruzar_poblacion(self, parejas: List[Tuple[int, int]]) -> np.ndarray:
        """
        Aplica cruza entre las parejas seleccionadas.

        Args:
            parejas: Lista de tuplas con los índices de los padres

        Returns:
            Nueva población tras la cruza
        """
        # Número de parejas
        n_parejas = len(parejas)

        # Crear una nueva población para los hijos
        tamano_poblacion_hijos = int(n_parejas * 2 * self.factor_crecimiento)
        poblacion_hijos = np.zeros((tamano_poblacion_hijos, self.bits), dtype=int)

        # Índice actual en la población de hijos
        idx_hijo = 0

        # Aplicar cruza a cada pareja
        for idx_padre1, idx_padre2 in parejas:
            padre1 = self.poblacion[idx_padre1]
            padre2 = self.poblacion[idx_padre2]

            # Realizar cruza de dos puntos
            hijo1, hijo2 = cruza_dos_puntos(padre1, padre2)

            # Añadir hijos a la nueva población
            if idx_hijo < tamano_poblacion_hijos:
                poblacion_hijos[idx_hijo] = hijo1
                idx_hijo += 1

            if idx_hijo < tamano_poblacion_hijos:
                poblacion_hijos[idx_hijo] = hijo2
                idx_hijo += 1

        return poblacion_hijos[:idx_hijo]  # Devolver solo los hijos generados

    def paso_generacion(self) -> Tuple[float, float, np.ndarray]:
        """
        Ejecuta un paso de evolución (una generación).

        Returns:
            Tupla con mejor fitness, fitness promedio y mejor individuo
        """
        # Evaluar población actual
        fitness = self._evaluar_poblacion()

        # Encontrar el mejor individuo y su fitness
        idx_mejor = np.argmax(fitness)
        mejor_individuo = self.poblacion[idx_mejor]
        mejor_fitness = fitness[idx_mejor]

        # Actualizar mejor solución global si corresponde
        if mejor_fitness > self.mejor_fitness:
            self.mejor_solucion = mejor_individuo.copy()
            self.mejor_fitness = mejor_fitness

        # Guardar estadísticas
        fitness_promedio = np.mean(fitness)
        self.mejor_fitness_historico.append(mejor_fitness)
        self.fitness_promedio_historico.append(fitness_promedio)

        # Valor real del mejor individuo
        mejor_valor_real = binario_a_real(
            mejor_individuo,
            self.rango_min,
            self.rango_max,
            self.bits
        )
        self.mejor_individuo_historico.append(mejor_valor_real)

        # Seleccionar parejas para cruza
        parejas = emparejamiento_aleatorio(self.poblacion)

        # Crear nueva población por cruza
        poblacion_hijos = self._cruzar_poblacion(parejas)

        # Aplicar mutación
        poblacion_hijos = mutacion_complemento(
            poblacion_hijos,
            self.tasa_mutacion_individuo,
            self.tasa_mutacion_gen
        )

        # Evaluar fitness de los hijos
        fitness_hijos = np.zeros(len(poblacion_hijos))
        for i, individuo in enumerate(poblacion_hijos):
            valor_real = binario_a_real(
                individuo,
                self.rango_min,
                self.rango_max,
                self.bits
            )
            fitness_hijos[i] = self.funcion_objetivo(valor_real)

        # Combinar poblaciones (padres + hijos)
        poblacion_combinada = np.vstack([self.poblacion, poblacion_hijos])
        fitness_combinado = np.concatenate([fitness, fitness_hijos])

        # Aplicar poda para volver al tamaño original
        self.poblacion, _ = poda_aleatoria_conservando_mejor(
            poblacion_combinada,
            fitness_combinado,
            self.tamano_poblacion
        )

        # Incrementar contador de generación
        self.generacion_actual += 1

        return mejor_fitness, fitness_promedio, mejor_individuo

    def evolucionar(self, pasos: int = None) -> Tuple[np.ndarray, float, float]:
        """
        Ejecuta el algoritmo genético durante un número de generaciones.

        Args:
            pasos: Número de pasos de evolución (si es None, usa max_generaciones)

        Returns:
            Mejor individuo encontrado, su valor de fitness y su valor real
        """
        if pasos is None:
            pasos = self.max_generaciones

        for _ in range(pasos):
            self.paso_generacion()

        # Convertir la mejor solución a valor real
        mejor_valor_real = binario_a_real(
            self.mejor_solucion,
            self.rango_min,
            self.rango_max,
            self.bits
        )

        return self.mejor_solucion, self.mejor_fitness, mejor_valor_real

    def obtener_estadisticas(self) -> dict:
        """
        Obtiene estadísticas del proceso evolutivo.

        Returns:
            Diccionario con estadísticas
        """
        return {
            'mejor_fitness_historico': self.mejor_fitness_historico,
            'fitness_promedio_historico': self.fitness_promedio_historico,
            'mejor_individuo_historico': self.mejor_individuo_historico,
            'generacion_actual': self.generacion_actual,
            'mejor_solucion_binaria': self.mejor_solucion,
            'mejor_fitness': self.mejor_fitness,
            'mejor_valor_real': binario_a_real(
                self.mejor_solucion,
                self.rango_min,
                self.rango_max,
                self.bits
            ) if self.mejor_solucion is not None else None
        }