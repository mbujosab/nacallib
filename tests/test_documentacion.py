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
    '*** Ejemplos de subespacios',
    '*** Ejemplos de espacios afines',
    '**** Ejemplos de presentación de las fracciones',
    '**** Ejemplos de diagonalización',
    '**** Ejemplos de transformaciones elementales',
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


def test_ejemplos_de_los_textos_de_ayuda():
    """Los ejemplos `>>>` de los textos de ayuda (help) muestran la salida real"""
    import doctest
    import nacal
    import nacal.nacal as modulo
    idioma_previo = nacal.idioma()
    try:
        resultado = doctest.testmod(modulo, verbose=False)
    finally:
        nacal.idioma(idioma_previo)
    assert resultado.attempted >= 70
    assert resultado.failed == 0


BLOQUE_CON_RESULTADO = re.compile(
    r'#\+BEGIN_SRC jupyter-python[^\n]*\n(.*?)#\+END_SRC\n(?:\n?#\+RESULTS:\n(.*?)\n(?=\n|\Z))?', re.S)


def no_se_exporta(texto, posicion):
    """Indica si la posición está en una sección COMMENT o :noexport:"""
    nivel = 99
    for cabecera in reversed(list(re.finditer(r'^(\*+) (.*)$', texto[:posicion], re.M))):
        if len(cabecera.group(1)) < nivel:
            nivel = len(cabecera.group(1))
            if cabecera.group(2).startswith('COMMENT') or ':noexport:' in cabecera.group(2):
                return True
    return False


def ejecuta_como_jupyter(codigo, entorno):
    """Ejecuta el bloque y escribe, como Jupyter, el valor de su última expresión"""
    import ast
    arbol = ast.parse(codigo)
    if arbol.body and isinstance(arbol.body[-1], ast.Expr):
        ultima = ast.Expression(arbol.body.pop().value)
        exec(compile(arbol, '<org>', 'exec'), entorno)
        valor = eval(compile(ultima, '<org>', 'eval'), entorno)
        if valor is not None:
            print(valor if isinstance(valor, str) else repr(valor))
    else:
        exec(codigo, entorno)


def test_todos_los_ejemplos_del_documento():
    """Todo bloque `jupyter-python` del documento con `#+RESULTS:` muestra la salida real

    Los bloques se ejecutan en el orden del documento y en un mismo entorno,
    como al evaluarlos en Emacs. Si esta prueba falla, hay que volver a
    ejecutar el bloque en Emacs (o corregir el resultado a mano).
    """
    import nacal
    import nacal.nacal as modulo

    def muestra(objeto, *args, **kwargs):
        # lo que Jupyter guarda como texto de un Math(...)
        datos = getattr(objeto, 'data', None)
        print('$\\displaystyle ' + datos + '$' if isinstance(datos, str) else repr(objeto))

    texto = ORG.read_text(encoding='utf-8')
    entorno = {}
    exec(PREAMBULO, entorno)
    entorno['display'] = muestra
    estado = nacal.idioma(), nacal.fracciones(), modulo.display
    modulo.display = muestra
    distintos, comprobados = [], 0
    try:
        for bloque in BLOQUE_CON_RESULTADO.finditer(texto):
            if no_se_exporta(texto, bloque.start()):
                continue
            codigo, resultado = bloque.groups()
            salida = io.StringIO()
            with contextlib.redirect_stdout(salida):
                ejecuta_como_jupyter(codigo, entorno)
            if resultado is None:
                continue
            esperado = resultado.strip()
            if esperado.startswith(':RESULTS:'):
                esperado = esperado[len(':RESULTS:'):].rsplit(':END:', 1)[0].strip()
            real = salida.getvalue().strip()
            comprobados += 1
            # en el documento, los saltos de línea de un texto se escriben \n
            if real != esperado and real != esperado.replace('\\n', '\n'):
                distintos.append('línea %d: %s' % (texto.count('\n', 0, bloque.start()) + 1,
                                                    codigo.strip().split('\n')[-1]))
    finally:
        modulo.display = estado[2]
        nacal.idioma(estado[0])
        nacal.fracciones(estado[1])
    assert comprobados >= 120
    assert not distintos, 'Resultados que no coinciden con la salida real:\n' + '\n'.join(distintos)
