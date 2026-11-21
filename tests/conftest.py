import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inspect
import pytest


def argumentos_por_defecto_mutables():
    """Lista (función, valor) de los argumentos por defecto mutables del módulo

    Muchas funciones del módulo tienen argumentos como `sust=[]`. Python crea
    esa lista una sola vez: si una función la modificara, el cambio se vería
    en todas las llamadas posteriores.
    """
    import nacal.nacal as modulo
    funciones = []
    for nombre, objeto in vars(modulo).items():
        if inspect.isfunction(objeto) and objeto.__module__ == modulo.__name__:
            funciones.append((nombre, objeto))
        elif inspect.isclass(objeto) and objeto.__module__ == modulo.__name__:
            for metodo, f in vars(objeto).items():
                f = getattr(f, '__func__', f)
                if inspect.isfunction(f):
                    funciones.append((nombre + '.' + metodo, f))
    mutables = []
    for nombre, f in funciones:
        valores = list(f.__defaults__ or ()) + list((f.__kwdefaults__ or {}).values())
        mutables += [(nombre, v) for v in valores if isinstance(v, (list, dict, set))]
    return mutables


@pytest.fixture(scope='session', autouse=True)
def los_argumentos_por_defecto_no_se_modifican():
    """Tras toda la batería, los argumentos por defecto mutables siguen vacíos"""
    yield
    modificados = [(nombre, valor) for nombre, valor in argumentos_por_defecto_mutables() if valor]
    assert not modificados, 'Alguna función ha modificado un argumento por defecto: %r' % modificados
