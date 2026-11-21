"""Presentación opcional de las fracciones en LaTeX."""
import pytest

from nacal import *

A = Matrix([[fracc(1, 2), 0], [fracc(1, 2), fracc(9, 10)]])
E = Matrix([[1, 2], [3, 4]])

def test_por_defecto_filas_separadas():
    """Desde la 0.5.5 el modo por defecto es 'separadas' (va antes de la preparación de abajo)"""
    import subprocess, sys
    codigo = ("from nacal import *; print(fracciones()); "
              "print(latex(Matrix([[fracc(1, 2), 0], [fracc(1, 2), fracc(9, 10)]]))); "
              "print(latex(Matrix([[1, 2], [3, 4]])))")
    salida = subprocess.run([sys.executable, '-c', codigo], capture_output=True, text=True, check=True,
                            cwd=str(__import__('pathlib').Path(__file__).resolve().parent.parent)).stdout.split('\n')
    assert salida[0] == 'separadas'
    assert salida[1] == r'\left[ \begin{array}{cc}\frac{1}{2}&0\\[0.6ex]\frac{1}{2}&\frac{9}{10}\\ \end{array} \right]'
    assert salida[2] == r'\left[ \begin{array}{cc}1&2\\3&4\\ \end{array} \right]'

@pytest.fixture(autouse=True)
def modo_normal():
    """Cada prueba empieza en modo 'normal' y al acabar se restaura el modo por defecto"""
    fracciones('normal')
    yield
    fracciones('separadas')

def test_modo_normal():
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
