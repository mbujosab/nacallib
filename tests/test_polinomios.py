"""Pruebas de la clase Polinomio y de su uso en la eliminación."""
import pytest
import sympy
from nacal import *

x, y, a = sympy.symbols('x y a')
P = Polinomio


def test_creacion():
    assert P((1 + x)**2).lista == [1, 2, 1]
    assert P([1, 2, 1]) == P((1 + x)**2) and P([1, 2, 1]).variable == x
    assert P(3).lista == [3] and P(0).lista == [0]
    assert P(0.5 + x).lista == [sympy.Rational(1, 2), 1]         # los decimales son racionales
    assert P(1 + y).variable == y
    assert P(a + x, variable=x).lista == [a, 1]
    assert P(a + x, variable=a).lista == [x, 1]
    assert type(P(1 + x)) is Polinomio and isinstance(P(1 + x), Vector)
    assert type(P(1 + x).copy()) is Polinomio and P(1 + y).copy().variable == y
    assert P(P(1 + y)).variable == y

def test_argumentos_incorrectos():
    with pytest.raises(ValueError):
        P(a + x)                    # dos símbolos y no se dice cuál es la variable
    with pytest.raises(ValueError):
        P(1/x)
    with pytest.raises(ValueError):
        P(sympy.sin(x))
    with pytest.raises(ValueError):
        P('hola')

def test_operaciones():
    assert 2*P(1 + x) + P(x**2) == P(2 + 2*x + x**2)
    assert P(1 + x) - P(1 + x + x**3) == P(-x**3)
    assert type(P(1) + P(x**2)) is Polinomio
    assert -P(1 - x) == P(x - 1)
    assert P(1 + x)*P(1 - x) == P(1 - x**2)
    assert P(1 + x)*3 == P(3 + 3*x) == 3*P(1 + x)
    assert P(1 + x) == P([1, 1, 0]) and P(1 + x) != P(1 + y) and P(1 + x) != Vector([1, 1])
    assert P(x) - P(x) == P(0) and (P(x) - P(x)).es_nulo()
    assert P(1 + x**3).grado() == 3 and P(5).grado() == 0 and P(0).grado() == -sympy.oo
    with pytest.raises(ValueError):
        P(1 + x) + P(1 + y)
    with pytest.raises(ValueError):
        P(1 + x) + Vector([1, 1])
    with pytest.raises(ValueError):
        P(1 + x)*Vector([1, 1])

def test_representacion():
    assert repr(P((1 + x)**2)) == 'Polinomio(1 + 2*x + x**2)'
    assert repr(P(-1 + x - x**3)) == 'Polinomio(-1 + x - x**3)'
    assert repr(P(0)) == 'Polinomio(0)' and str(P(1 - x)) == '1 - x'
    assert latex(P(3 - 2*x + 5*x**2)) == '3 - 2 x + 5 x^{2}'
    assert latex(P(x/2 - x**2)) == r'\frac{x}{2} - x^{2}'
    assert repr(P(a*x**2 - (a + 1)*x, variable=x)) == 'Polinomio((-a - 1)*x + a*x**2)'
    assert latex(P(a*x**2 - (a + 1)*x, variable=x)) == r'\left(- a - 1\right) x + a x^{2}'

def test_los_vectores_no_cambian():
    assert type(Vector([1, 2]) + Vector([1, 1])) is Vector
    assert Vector([1, 2])*Vector([1, 1]) == 3
    with pytest.raises(ValueError):
        Vector([1, 2]) + Vector([1, 2, 3])

def test_SEL_con_polinomios():
    S = Sistema([P(1), P(1 + x), P((1 + x)**2)])
    q = P(3 - 2*x + 5*x**2)
    sol = SEL(S, q)
    assert sol.solP == Vector([10, -12, 5]) and sol.determinado
    assert S*sol.solP == q
    assert sol.tex.startswith(r'\left[ \begin{array}{ccc|c}1&1 + x&1 + 2 x + x^{2}&-3 + 2 x - 5 x^{2}\\')
    assert r'\boldsymbol{v}=3 - 2 x + 5 x^{2}' in sol.latex()
    assert SEL(Sistema([P(1), P(x)]), P(x**2)).solP == set()
    assert desarrollo(sol, por_linea=1).count('[1ex]') == 2

def test_Homogenea_y_Elim_con_polinomios():
    S = Sistema([P(1 + x), P(1 - x), P(x), P(x**2)])
    H = Homogenea(S)
    assert H.sgen == Sistema([Vector([-1, 1, 2, 0])]) and not H.determinado
    assert (S*(H.sgen|1)).es_nulo()
    K = Elim(S)
    assert all(type(p) is Polinomio for p in K)
    assert [p.es_nulo() for p in K] == [False, False, True, False]
    assert Homogenea(Sistema([P(1), P(x), P(x**2)])).determinado

def test_un_sistema_de_vectores_sigue_dando_una_matriz_de_coeficientes():
    S = Sistema([Vector([1, 0]), Vector([1, 1])])
    assert SEL(S, Vector([2, 3])).tex == SEL(Matrix(S), Vector([2, 3])).tex
