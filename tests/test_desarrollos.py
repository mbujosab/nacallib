"""Los desarrollos recuerdan sus eslabones (etapa 1 de la revisión de coherencia)."""
import copy

import pytest
import sympy

from nacal import *
from nacal.nacal import TexDesarrollo, trozos_del_desarrollo

A = Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]])
B = Matrix([[1, 2, 0], [2, 4, 1], [1, 1, 1], [0, 3, 2]])
a = sympy.symbols('a')
C = Matrix([[1, a], [a, 1]])

CASOS = {
    'elim':        lambda: A.elim(),
    'encadenada':  lambda: A.elim(20).elim(10),
    'Elim':        lambda: Elim(B),
    'ElimG':       lambda: ElimG(A),
    'ElimGJ':      lambda: ElimGJ(A),
    'ElimF':       lambda: ElimGF(B),
    'InvMat':      lambda: InvMat(A),
    'InvMatF':     lambda: InvMatF(A),
    'InvMatFC':    lambda: InvMatFC(A),
    'Homogenea':   lambda: Homogenea(B.apila(B)),
    'SEL':         lambda: SEL(A, Vector([2, -9, 2])),
    'simbolica':   lambda: Elim(C),
    'diagonalizaC': lambda: Matrix([[2, 1], [1, 2]]).diagonalizaC(),
    'diagonalizaS': lambda: Matrix([[2, 1], [1, 2]]).diagonalizaS([1, 3]),
    'Determinante': lambda: Determinante(A),
}

@pytest.mark.parametrize('nombre', sorted(CASOS))
def test_los_eslabones_guardados_son_los_del_codigo(nombre):
    tex = CASOS[nombre]().tex
    assert isinstance(tex, str)
    trozos = tex.trozos if isinstance(tex, TexDesarrollo) else trozos_del_desarrollo(tex)
    assert trozos == trozos_del_desarrollo(str(tex))
    assert ''.join(trozos) == tex
    assert len(trozos) % 2 == 1

@pytest.mark.parametrize('nombre', ['elim', 'encadenada', 'Elim', 'ElimGJ', 'InvMat', 'SEL', 'Homogenea'])
def test_las_funciones_de_eliminacion_guardan_los_eslabones(nombre):
    assert isinstance(CASOS[nombre]().tex, TexDesarrollo)

@pytest.mark.parametrize('nombre', sorted(CASOS))
def test_desarrollo_da_lo_mismo_con_el_objeto_que_con_la_cadena(nombre):
    objeto = CASOS[nombre]()
    cadena = str(objeto.tex)
    for opciones in ({}, {'por_linea': 2}, {'pasos': []}, {'pasos': [1], 'omitidos': 'puntos'},
                     {'por_linea': 1, 'compacto': True}):
        if 'pasos' in opciones and opciones['pasos'] and len(trozos_del_desarrollo(cadena)) < 3:
            continue
        assert desarrollo(objeto, **opciones) == desarrollo(cadena, **opciones)

def test_continuar_un_codigo_escrito_a_mano():
    previo = r'X\xrightarrow{a}Y'
    tex = TexDesarrollo(previo, [r'\xrightarrow{b}', 'Z'])
    assert tex == r'X\xrightarrow{a}Y\xrightarrow{b}Z'
    assert tex.trozos == ['X', r'\xrightarrow{a}', 'Y', r'\xrightarrow{b}', 'Z']

def test_se_puede_copiar_y_concatenar():
    tex = ElimGJ(A).tex
    copia = copy.deepcopy(tex)
    assert copia == tex and copia.trozos == tex.trozos
    assert copy.deepcopy(ElimGJ(A)).tex == tex
    assert '$' + tex + '$' == '$' + str(tex) + '$'
