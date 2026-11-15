"""Los ejemplos de la documentación (nacal_src.org) deben mostrar la salida real.

Se ejecutan, en orden, los bloques `jupyter-python` de las secciones indicadas
en SECCIONES, y se compara lo que imprimen con el bloque `#+RESULTS:` que les
sigue en el documento.
"""
import contextlib
import io
import pathlib
import re

import pytest

ORG = pathlib.Path(__file__).resolve().parent.parent / 'nacal_src.org'

# Encabezados de las secciones cuyos ejemplos se comprueban
SECCIONES = [
    '**** Ejemplos de eliminación',
    '***** Ejemplos de matrices inversas',
    '*** Ejemplos de sistemas homogéneos',
    '*** Ejemplos de sistemas de ecuaciones',
    '*** Ejemplo: combinaciones lineales de matrices',
    '*** Ejemplo: combinaciones lineales de polinomios',
    '*** Ejemplos de determinantes',
]

PREAMBULO = '''
from nacal import *
def displaymath(t): return '$$' + t + '$$'
def dmath(t):       return '$$' + latex(t) + '$$'
def math(t):        return '$' + latex(t) + '$'
def dprint(t):      return '=' + str(t) + '='
a,b,c,x,y,z = sympy.symbols('a b c x y z')
'''

BLOQUE = re.compile(r'#\+BEGIN_SRC jupyter-python[^\n]*\n(.*?)#\+END_SRC\n(?:\n#\+RESULTS:\n(.*?)\n(?=\n|\Z))?', re.S)


def seccion(encabezado):
    texto = ORG.read_text(encoding='utf-8')
    assert texto.count('\n' + encabezado + '\n') == 1, encabezado
    inicio = texto.index('\n' + encabezado + '\n') + 1
    nivel = len(encabezado.split(' ')[0])
    fin = re.compile(r'^\*{1,%d} ' % nivel, re.M).search(texto, inicio + len(encabezado))
    return texto[inicio: fin.start() if fin else len(texto)]


@pytest.mark.parametrize('encabezado', SECCIONES)
def test_ejemplos_de_la_documentacion(encabezado):
    entorno = {}
    exec(PREAMBULO, entorno)
    comprobados = 0
    for codigo, resultado in BLOQUE.findall(seccion(encabezado)):
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            exec(codigo, entorno)
        if resultado:
            assert salida.getvalue().strip() == resultado.strip(), codigo
            comprobados += 1
    assert comprobados > 0
