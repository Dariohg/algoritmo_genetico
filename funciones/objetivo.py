import numpy as np


def funcion_objetivo(x):
    """
    Función objetivo a maximizar:
    f(x) = ln(10 + 3 cos(7x) - 5 sen(13x) + abs(x))

    Args:
        x: Valor(es) de entrada

    Returns:
        Valor(es) de la función
    """
    # Asegurarse que el argumento del logaritmo sea positivo
    argumento = 10 + 3 * np.cos(7 * x) - 5 * np.sin(13 * x) + np.abs(x)

    # Si algún valor es menor o igual a cero, retornar un valor muy negativo
    if np.any(argumento <= 0):
        if np.isscalar(argumento):
            return -np.inf
        else:
            resultado = np.log(np.maximum(argumento, 1e-10))
            resultado[argumento <= 0] = -np.inf
            return resultado

    return np.log(argumento)