"""Representación Python de SubEspacio y EAfin (etapa 3 de la revisión de coherencia)."""
from nacal import *

CASOS = [
    SubEspacio(Sistema([Vector([1, 0, 1]), Vector([0, 1, 1])])),
    SubEspacio(Sistema([Vector([0, 0, 0])])),
    SubEspacio(Matrix([[1, 1, 0], [0, 1, 1]])),
    Matrix([[1, 2], [2, 4]]).espacio_nulo(),
    EAfin(Sistema([Vector([1, 0, 1])]), Vector([1, 2, 3])),
    EAfin(Sistema([Vector([1, 0, 1])]), Vector([1, 2, 3]), 1),
    EAfin(Sistema([Vector([0, 0])]), Vector([1, 2])),
    SEL(Matrix([[1, 1, 0], [0, 1, 1]]), Vector([1, 2])).eafin,
]

def test_repr_reconstruye_el_objeto():
    for objeto in CASOS:
        texto = repr(objeto)
        assert texto.startswith(type(objeto).__name__ + '(')
        copia = eval(texto)
        assert copia == objeto
        assert copia.latex() == objeto.latex()
        assert repr(copia) == texto

def test_ejemplos():
    assert repr(CASOS[0]) == 'SubEspacio(Sistema([Vector([1, 0, 1]), Vector([0, 1, 1])]))'
    assert repr(CASOS[5]) == 'EAfin(Sistema([Vector([1, 0, 1])]), Vector([1, 2, 3]), 1)'
