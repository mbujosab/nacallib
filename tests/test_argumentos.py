"""Argumentos uniformes entre funciones equivalentes (etapa 2 de la revisión de coherencia)."""
import sympy

from nacal import *

a = sympy.symbols('a')
A = Matrix([[1, a], [0, 2]])
N = Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]])

def test_inversa_admite_sust_como_InvMat():
    regla = [(a, 3)]
    por_metodo, por_clase = A.inversa(sust=regla), InvMat(A, sust=regla)
    assert por_metodo == por_clase == Matrix([[1, 3], [0, 2]]).inversa()
    assert por_metodo.tex == por_clase.tex
    assert A.inversa(0, regla, 1).tex == InvMat(A, 0, regla, 1).tex
    assert type(por_metodo) is Matrix

def test_inversa_sin_sust_no_cambia():
    assert N.inversa() == InvMat(N)
    assert N.inversa().tex == InvMat(N).tex

def test_Determinante_admite_rep_ademas_de_disp():
    assert Determinante(N, rep=0).valor == Determinante(N, disp=0).valor == Determinante(N).valor
    assert Determinante(N, rep=0).tex == Determinante(N).tex
    assert Determinante(A, 0, [(a, 3)]).valor == Determinante(A, sust=[(a, 3)], rep=0).valor == 2

def test_DiagonalizaS_admite_sust():
    B = Matrix([[2, 1], [1, a]])
    D = DiagonalizaS(B, [1, 3], sust=[(a, 2)])
    E = DiagonalizaS(Matrix([[2, 1], [1, 2]]), [1, 3])
    assert D == E and D.S == E.S
    assert D == B.diagonalizaS([1, 3], sust=[(a, 2)])
