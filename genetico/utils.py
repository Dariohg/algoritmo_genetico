import numpy as np
from typing import Tuple, List


def contar_bits_valor_real(rango_min: float, rango_max: float, precision: float) -> int:
    """
    Calcula la cantidad de bits necesarios para representar un rango con cierta precisión.

    Args:
        rango_min: Valor mínimo del rango
        rango_max: Valor máximo del rango
        precision: Precisión deseada

    Returns:
        Número de bits necesarios
    """
    valores_posibles = (rango_max - rango_min) / precision + 1
    bits_necesarios = int(np.ceil(np.log2(valores_posibles)))
    return bits_necesarios


def decimal_a_binario(valor_decimal: int, bits: int) -> np.ndarray:
    """
    Convierte un valor decimal a su representación binaria.

    Args:
        valor_decimal: Valor decimal a convertir
        bits: Número de bits para la representación

    Returns:
        Array con la representación binaria
    """
    binario = np.zeros(bits, dtype=int)

    for i in range(bits):
        if valor_decimal & (1 << i):
            binario[bits - 1 - i] = 1

    return binario


def real_a_binario(valor_real: float, rango_min: float, rango_max: float, bits: int) -> np.ndarray:
    """
    Convierte un valor real a su representación binaria.

    Args:
        valor_real: Valor real a convertir
        rango_min: Valor mínimo del rango
        rango_max: Valor máximo del rango
        bits: Número de bits para la representación

    Returns:
        Array con la representación binaria
    """
    # Normalizar el valor real al rango [0, 1]
    valor_normalizado = (valor_real - rango_min) / (rango_max - rango_min)

    # Convertir a valor decimal
    max_decimal = 2 ** bits - 1
    valor_decimal = int(valor_normalizado * max_decimal)

    # Convertir a binario
    return decimal_a_binario(valor_decimal, bits)


def binario_a_decimal(individuo_binario: np.ndarray) -> int:
    """
    Convierte un individuo binario a su valor decimal.

    Args:
        individuo_binario: Array con la representación binaria

    Returns:
        Valor decimal
    """
    valor_decimal = 0
    for i, bit in enumerate(reversed(individuo_binario)):
        valor_decimal += bit * (2 ** i)
    return valor_decimal


def binario_a_real(individuo_binario: np.ndarray, rango_min: float, rango_max: float, bits: int) -> float:
    """
    Convierte un individuo binario a su valor real en el rango especificado.

    Args:
        individuo_binario: Array con la representación binaria
        rango_min: Valor mínimo del rango
        rango_max: Valor máximo del rango
        bits: Número de bits usados para la codificación

    Returns:
        Valor real
    """
    valor_decimal = binario_a_decimal(individuo_binario)
    max_decimal = 2 ** bits - 1
    valor_real = rango_min + (valor_decimal / max_decimal) * (rango_max - rango_min)
    return valor_real


def imprimir_poblacion_info(poblacion: np.ndarray, fitness: np.ndarray, rango_min: float, rango_max: float,
                            bits: int) -> None:
    """
    Imprime información detallada sobre la población.

    Args:
        poblacion: Población binaria
        fitness: Valores de fitness de la población
        rango_min: Valor mínimo del rango
        rango_max: Valor máximo del rango
        bits: Número de bits usados para la codificación
    """
    print("\nInformación de la población:")
    print("----------------------------")
    print(f"Tamaño de población: {len(poblacion)}")
    print(f"Longitud de individuo: {bits} bits")

    # Ordenar por fitness (de mayor a menor para maximización)
    indices_ordenados = np.argsort(-fitness)

    # Mostrar los 5 mejores individuos
    print("\nMejores individuos:")
    for i in range(min(5, len(poblacion))):
        idx = indices_ordenados[i]
        binario = ''.join(map(str, poblacion[idx]))
        valor_real = binario_a_real(poblacion[idx], rango_min, rango_max, bits)
        print(f"{i + 1}. Binario: {binario}, Valor real: {valor_real:.6f}, Fitness: {fitness[idx]:.6f}")

    # Estadísticas de fitness
    print("\nEstadísticas de fitness:")
    print(f"Máximo: {np.max(fitness):.6f}")
    print(f"Mínimo: {np.min(fitness):.6f}")
    print(f"Promedio: {np.mean(fitness):.6f}")
    print(f"Desviación estándar: {np.std(fitness):.6f}")


def calcular_diversidad_hamming(poblacion: np.ndarray) -> float:
    """
    Calcula la diversidad de la población usando la distancia de Hamming.

    Args:
        poblacion: Población binaria

    Returns:
        Diversidad promedio (distancia de Hamming promedio entre pares de individuos)
    """
    n_individuos = len(poblacion)

    if n_individuos <= 1:
        return 0.0

    distancia_total = 0
    pares_comparados = 0

    for i in range(n_individuos):
        for j in range(i + 1, n_individuos):
            # Distancia de Hamming (número de bits diferentes)
            distancia = np.sum(poblacion[i] != poblacion[j])
            distancia_total += distancia
            pares_comparados += 1

    return distancia_total / pares_comparados if pares_comparados > 0 else 0.0


def calcular_estadisticas_convergencia(
        mejor_fitness_historico: List[float],
        fitness_promedio_historico: List[float]
) -> dict:
    """
    Calcula estadísticas para analizar la convergencia del algoritmo.

    Args:
        mejor_fitness_historico: Lista con el mejor fitness de cada generación
        fitness_promedio_historico: Lista con el fitness promedio de cada generación

    Returns:
        Diccionario con estadísticas de convergencia
    """
    # Verificar que haya suficientes datos
    if len(mejor_fitness_historico) < 2:
        return {
            'convergencia_rapida': False,
            'generacion_convergencia': -1,
            'tasa_mejora_temprana': 0.0,
            'tasa_mejora_tardia': 0.0,
            'estancamiento': False
        }

    # Calcular diferencias entre generaciones consecutivas
    diferencias = np.diff(mejor_fitness_historico)

    # Detectar convergencia (cuando las mejoras son muy pequeñas)
    umbral_convergencia = 1e-6
    gen_convergencia = -1

    for i, diff in enumerate(diferencias):
        if abs(diff) < umbral_convergencia:
            # Verificar si las siguientes 5 generaciones también muestran poca mejora
            if i + 5 < len(diferencias) and all(abs(d) < umbral_convergencia for d in diferencias[i:i + 5]):
                gen_convergencia = i
                break

    # Calcular tasas de mejora
    n_gen = len(mejor_fitness_historico)
    mitad = n_gen // 2

    if mitad > 0:
        tasa_mejora_temprana = (mejor_fitness_historico[mitad] - mejor_fitness_historico[0]) / mitad if mitad > 0 else 0
    else:
        tasa_mejora_temprana = 0

    if n_gen - mitad > 0:
        tasa_mejora_tardia = (mejor_fitness_historico[-1] - mejor_fitness_historico[mitad]) / (
                    n_gen - mitad) if n_gen - mitad > 0 else 0
    else:
        tasa_mejora_tardia = 0

    # Detectar estancamiento (cuando hay poca mejora en la última parte)
    ultimas_gen = min(20, n_gen // 4)
    estancamiento = False

    if ultimas_gen > 0 and n_gen > ultimas_gen:
        mejora_reciente = mejor_fitness_historico[-1] - mejor_fitness_historico[-ultimas_gen]
        estancamiento = mejora_reciente < umbral_convergencia * ultimas_gen

    return {
        'convergencia_rapida': gen_convergencia < n_gen // 3 and gen_convergencia != -1,
        'generacion_convergencia': gen_convergencia,
        'tasa_mejora_temprana': tasa_mejora_temprana,
        'tasa_mejora_tardia': tasa_mejora_tardia,
        'estancamiento': estancamiento
    }