import numpy as np


def calcular_bits_necesarios(rango_min, rango_max, precision):
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


def inicializar_poblacion_binaria(tamano_poblacion, longitud_individuo):
    """
    Crea una población inicial aleatoria en representación binaria.

    Args:
        tamano_poblacion: Número de individuos
        longitud_individuo: Longitud en bits de cada individuo

    Returns:
        Población inicial binaria (numpy array)
    """
    return np.random.randint(2, size=(tamano_poblacion, longitud_individuo))


def binario_a_decimal(individuo_binario):
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


def binario_a_real(individuo_binario, rango_min, rango_max, bits):
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