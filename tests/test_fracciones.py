"""Presentación opcional de las fracciones en LaTeX."""
import pytest

from nacal import *

A = Matrix([[fracc(1, 2), 0], [fracc(1, 2), fracc(9, 10)]])
E = Matrix([[1, 2], [3, 4]])

@pytest.fixture(autouse=True)
def modo_normal():
    fracciones('normal')
    yield
    fracciones('normal')

def test_por_defecto_nada_cambia():
    assert fracciones() == 'normal'
    assert latex(A) == r'\left[ \begin{array}{cc}\frac{1}{2}&0\\\frac{1}{2}&\frac{9}{10}\\ \end{array} \right]'

def test_separadas():
    normal = latex(A), latex(E), latex(Vector([fracc(1, 2), 1])), latex(Vector([1, 2]))
    fracciones('separadas')
    assert latex(A) == r'\left[ \begin{array}{cc}\frac{1}{2}&0\\[0.6ex]\frac{1}{2}&\frac{9}{10}\\ \end{array} \right]'
    assert latex(E) == normal[1]                        # sin fracciones no cambia
    assert latex(Vector([1, 2])) == normal[3]
    assert latex(Vector([fracc(1, 2), 1])) == r'\begin{pmatrix}\frac{1}{2}\\[0.6ex] 1\end{pmatrix}'
    assert latex(A).replace('[0.6ex]', '') == normal[0]
    fracciones('separadas', '1ex')
    assert r'\\[1ex]' in latex(A)

def test_separadas_con_lineas_de_corte():
    B = A.apila(I(2), 1)
    normal = latex(B)
    fracciones('separadas')
    assert latex(B).replace('[0.6ex]', '') == normal
    assert r'\\[0.6ex] \hline ' in latex(B)

def test_inclinadas():
    fracciones('inclinadas')
    assert latex(A) == r'\left[ \begin{array}{cc}1 / 2&0\\1 / 2&9 / 10\\ \end{array} \right]'
    assert r'\frac' not in ElimGJ(A).tex

def test_los_desarrollos_se_pueden_recomponer_en_cualquier_modo():
    for modo in ('separadas', 'inclinadas'):
        fracciones(modo)
        tex = InvMat(A).tex
        assert desarrollo(tex) == tex
        assert desarrollo(str(tex), por_linea=1) == desarrollo(InvMat(A), por_linea=1)

def test_modo_desconocido():
    with pytest.raises(ValueError):
        fracciones('decimales')
