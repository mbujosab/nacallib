"""Pruebas de los fallos corregidos en noviembre de 2026."""
import warnings
import pytest
import sympy
from nacal import *

r2 = sympy.sqrt(2)
x, a = sympy.symbols('x a')


# --- detección de ceros "disfrazados"
def test_es_cero():
    assert es_cero(0) and es_cero(0.0) and es_cero(sympy.S(0))
    assert es_cero(sympy.sin(x)**2 + sympy.cos(x)**2 - 1)
    assert es_cero(sympy.Add(-3, -2*r2, (1 + r2)**2, evaluate=False))
    assert es_cero((a + 1)**2 - a**2 - 2*a - 1)
    assert not es_cero(a) and not es_cero(r2) and not es_cero(a**2 - 1)
    assert not es_cero(sympy.pi - 3) and not es_cero(1/a)

def test_rango_con_radicales():
    A = Matrix([[0, 1], [(1 + r2)**2 - 3 - 2*r2, 1]])
    assert A.rango() == 1
    assert A.K() == Matrix([[0, 1], [0, 1]])

def test_rango_con_identidad_trigonometrica():
    B = Matrix([[sympy.sin(x)**2 + sympy.cos(x)**2, 1], [1, 1]])
    assert B.rango() == 1
    assert B.es_singular()
    with pytest.raises(ValueError):
        B.inversa()

def test_es_nulo():
    assert Vector([0, sympy.sin(x)**2 + sympy.cos(x)**2 - 1]).es_nulo()
    assert Vector([0, a]).no_es_nulo()
    assert Vector([0, a]).es_nulo([(a, 0)])
    assert M0(2, 3).es_nulo() and I(2).no_es_nulo()
    assert Sistema([]).es_nulo()
    assert Sistema([V0(2), M0(2)]).es_nulo()


# --- comprobaciones que no se ejecutaban
def test_potencia_de_matriz_no_cuadrada():
    with pytest.raises(ValueError, match='cuadrada'):
        Matrix([[1, 2, 3], [4, 5, 6]]) ** 2

def test_diagonalizaS_de_matriz_no_cuadrada():
    with pytest.raises(ValueError, match='cuadrada'):
        Matrix([[1, 2, 3], [4, 5, 6]]).diagonalizaS([1, 2])

def test_variante_de_eliminacion_no_valida():
    A = Matrix([[1, 2], [3, 4]])
    for variante in (8, 50, 3.5, -1):
        with pytest.raises(ValueError):
            A.elim(variante)
    for variante in (0, 1, 2, 3, 4, 7, 10, 14, 20, 24, 26, 32, 36, 100, 120, 137):
        A.elim(variante)


# --- igualdad y productos con objetos de otro tipo
def test_igualdad_con_otros_tipos():
    v = Vector([1, 2])
    assert not v == 3
    assert not v == None
    assert v != [1, 2]
    assert v == Sistema([1, 2])
    assert not T({1, 2}) == 3
    assert T({1, 2}) == T({1, 2})

def test_producto_no_definido_lanza_error():
    A = Matrix([[1, 2], [3, 4]])
    with pytest.raises(TypeError):
        A * 'x'
    with pytest.raises(TypeError):
        'x' * A
    with pytest.raises(TypeError):
        A * BlockV(['a', 'b'])
    assert a * A == A * a


# --- espacios afines
def test_interseccion_de_espacios_afines():
    v, w = Vector([1, 2, 3]), Vector([2, 0, -1])
    E1 = EAfin(Sistema([v, w]), Vector([1, 1, 1]))
    E2 = EAfin(Sistema([v, Vector([0, 0, 1])]), Vector([0, 1, 0]))
    F  = E1 & E2
    assert F.S.dim == 1
    assert F.v in E1 and F.v in E2
    assert F.contenido_en(E1) and F.contenido_en(E2)
    G = E1 & SubEspacio(Sistema([v, Vector([0, 0, 1])]))
    assert G.contenido_en(E1)

def test_pertenencia_a_EAfin_de_matrices_no_cuadradas():
    M = Matrix([[1, 2, 3], [4, 5, 6]])
    E = EAfin(Sistema([M]), M0(2, 3))
    assert M in E
    assert M1(2, 3) not in E


# --- funciones lineales
def test_imagen_de_un_punto_fuera_del_dominio():
    f = FuncionLineal([Vector([1, 0, 0])], [Vector([1, 1])])
    assert f.imagen_de(Vector([2, 0, 0])) == Vector([2, 2])
    with pytest.raises(ValueError, match='dominio'):
        f.imagen_de(Vector([0, 1, 0]))
    with pytest.raises(ValueError, match='imagen'):
        f.preimagen_de(Vector([1, 0]))


# --- compatibilidad con versiones futuras de Python
def test_el_modulo_compila_sin_avisos():
    import nacal.nacal as modulo
    with open(modulo.__file__, encoding='utf8') as f:
        fuente = f.read()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        compile(fuente, modulo.__file__, 'exec')


# --- concatenación de vectores
def test_concatenar_vectores_da_un_vector():
    v = Vector([1, 2]).concatena(Vector([3]))
    assert isinstance(v, Vector) and v == Vector([1, 2, 3]) and v.n == 3
    assert isinstance(Vector([1, 2]).amplia(3), Vector)
    assert isinstance(Vector([1, 2]).amplia([3, 4]), Vector)
    assert Vector([1, 2]).concatena(Vector([3]), 1).corteSistema == {2}
    # si lo añadido no son números el resultado es un Sistema genérico
    assert type(Vector([1, 2]).concatena(Sistema(['a']))) is Sistema
    assert type(Vector([1, 2]).concatena(Sistema([Vector([1])]))) is Sistema

def test_latex_no_oculta_errores():
    class Roto:
        def latex(self):
            raise RuntimeError('fallo al representar')
    with pytest.raises(RuntimeError):
        latex(Roto())
    assert latex(sympy.Rational(1, 2)) == r'\frac{1}{2}'
    assert latex('hola') == sympy.latex('hola')


# --- los float se convierten en racionales exactos
def test_float_a_racional():
    R = sympy.Rational
    assert racional(0.1) == R(1, 10) and racional(-0.75) == R(-3, 4) and racional(2.0) == 2
    assert racional(3) == 3 and racional('a') == 'a' and racional(a) == a
    assert racional(0.5*a + 0.25) == a/2 + R(1, 4)
    assert Vector([0.5, 1]).lista == [R(1, 2), 1]
    assert (0.5 * Vector([1, 2])) == Vector([R(1, 2), 1])
    assert (Vector([1, 2]) * 0.1) == Vector([R(1, 10), R(1, 5)])
    assert Vector([a, 1]).subs([(a, 0.5)]) == Vector([R(1, 2), 1])
    assert T((0.5, 1)).abreviaturas == [(R(1, 2), 1)]
    assert fracc(0.5, 2) == R(1, 4)

def test_rango_con_decimales():
    A = Matrix([[0.1, 0.2, 0.3], [0.2, 0.4, 0.6], [0.3, 0.6, 0.9]])
    assert A.rango() == 1
    assert Matrix([[0.5, 1], [1, 1]]).inversa() == Matrix([[-2, 2], [2, -1]])
    assert SEL(Matrix([[0.5, 1], [1, 1]]), Vector([1, 2])).solP == Vector([2, 0])


# --- presentación de las soluciones de un sistema de matrices
def test_latex_de_SEL():
    S = Sistema([Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]]),
                 Matrix([[1, 1], [1, 0]]), Matrix([[0, 0], [0, 1]])])
    B = Matrix([[2, 3], [3, 5]])
    s = SEL(S, B)
    assert s.latex() == s.EcCombinacion() + r'\; = \;' + s.eafin.EcParametricas()
    assert latex(S) + r'\boldsymbol{v}=' + latex(B) in s.latex()
    # determinado
    d = SEL(S | (1, 2, 4), B)
    assert d.determinado and d.latex().startswith(d.EcCombinacion())
    # incompatible
    assert SEL(S, Matrix([[2, 3], [4, 5]])).latex() == r'\emptyset'
    # con vectores la representación no cambia
    A = Matrix([[1, 2], [2, 4]])
    assert SEL(A, Vector([1, 2])).latex() == SEL(A, Vector([1, 2])).eafin.EcParametricas()
    assert SEL(Sistema([Vector([1, 2]), Vector([2, 4])]), Vector([1, 2])).latex() == SEL(A, Vector([1, 2])).latex()
    assert SEL(I(2), Vector([1, 2])).latex() == r'\left\{\ ' + latex(Vector([1, 2])) + r'\ \right\}'
