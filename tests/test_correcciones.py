"""Pruebas de los fallos corregidos en noviembre de 2026."""
import warnings
import pytest
import sympy
from nacal import *
from nacal.nacal import trozos_del_desarrollo, TEXTOS_DE_SALIDA   # no se exportan

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


# --- presentación de los desarrollos: varias líneas, compactos o abreviados
def _desarrollos():
    a, b, c, d = sympy.symbols('a b c d')
    A = Matrix([[a, b], [c, d]])
    M = Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]])
    return [SEL(Sistema([A**0, A]), A**2), ElimGJ(M), ElimGJF(M), InvMat(M),
            Determinante(M), Homogenea(Matrix([[1, 2, 3], [2, 4, 6]])), ElimG(Elim(M))]

def test_desarrollo_por_defecto_no_cambia_nada():
    for objeto in _desarrollos():
        trozos = trozos_del_desarrollo(objeto.tex)
        assert ''.join(trozos) == objeto.tex
        assert desarrollo(objeto) == desarrollo(objeto.tex) == objeto.tex
        assert len(trozos) == 2*objeto.tex.count('xrightarrow') + 1
        assert all(f.startswith('\\xrightarrow') for f in trozos[1::2])
        assert not any('xrightarrow' in s for s in trozos[0::2])

def test_desarrollo_por_lineas():
    E = ElimGJ(Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]]))     # 5 pasos: 6 matrices
    for por_linea, lineas in [(1, 6), (2, 3), (3, 2), (4, 2), (5, 2), (6, 1), (9, 1)]:
        tex = desarrollo(E, por_linea=por_linea)
        assert tex.count('\\\\[1ex]') == lineas - 1
        if lineas > 1:      # todas las líneas llevan por_linea matrices, salvo quizá la última
            cuerpo = tex.replace('\\begin{aligned}{}&', '').replace('\\end{aligned}', '')
            flechas = [l.count('xrightarrow') for l in cuerpo.split('\\\\[1ex]&')]
            assert flechas[0] == por_linea - 1 and all(f == por_linea for f in flechas[1:-1])
        assert tex.replace('\\begin{aligned}{}&', '').replace('\\\\[1ex]&', '').replace('\\end{aligned}', '') == E.tex
    with pytest.raises(ValueError):
        desarrollo(E, por_linea=0)

def test_desarrollo_compacto():
    E = ElimGJ(Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]]))
    assert desarrollo(E, compacto=True) == '{\\setlength{\\arraycolsep}{2pt}' + E.tex + '}'
    assert desarrollo(E, compacto=1.5).startswith('{\\setlength{\\arraycolsep}{1.5pt}')

def test_desarrollo_abreviado():
    E = ElimGJ(Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]]))     # 5 pasos
    sistemas, flechas = trozos_del_desarrollo(E.tex)[0::2], trozos_del_desarrollo(E.tex)[1::2]
    puntos = '\\xrightarrow{\\ \\cdots\\ }'
    assert desarrollo(E, pasos=[], omitidos='puntos') == sistemas[0] + puntos + sistemas[-1]
    assert desarrollo(E, pasos=[1, -2], omitidos='puntos') == sistemas[0] + flechas[0] + sistemas[1] + puntos + sistemas[4] + flechas[4] + sistemas[5]
    assert desarrollo(E, pasos=[1, 2, 3, 4]) == E.tex
    with pytest.raises(ValueError):
        desarrollo(E, pasos=[6])

def test_desarrollo_abreviado_con_transformaciones():
    M = Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]])
    E = ElimGJ(M)                                               # 5 pasos por la derecha
    sistemas, flechas = trozos_del_desarrollo(E.tex)[0::2], trozos_del_desarrollo(E.tex)[1::2]
    rotulos = [f[len('\\xrightarrow{'):-1] for f in flechas]
    assert desarrollo(E, pasos=[]) == sistemas[0] + '\\xrightarrow{' + ''.join(rotulos) + '}' + sistemas[-1]
    assert desarrollo(E, pasos=[2]) == sistemas[0] + '\\xrightarrow{' + ''.join(rotulos[:2]) + '}' + sistemas[2] \
                                                   + '\\xrightarrow{' + ''.join(rotulos[2:]) + '}' + sistemas[5]
    F = ElimGJF(M)                                              # pasos por la izquierda: orden inverso
    flechasF = trozos_del_desarrollo(F.tex)[1::2]
    inferiores = [f[len('\\xrightarrow['):-len(']{}')] for f in flechasF]
    assert all(f.endswith(']{}') for f in flechasF)
    assert desarrollo(F, pasos=[]).count('\\xrightarrow[' + ''.join(reversed(inferiores)) + ']{}') == 1
    D = Determinante(M)                                         # rótulos arriba y abajo
    assert desarrollo(D, pasos=[1, 2, 3]) == D.tex
    with pytest.raises(ValueError):
        desarrollo(E, pasos=[], omitidos='nada')


# --- idioma de los textos de salida
def test_idioma():
    A, b = Matrix([[1, 2], [2, 4]]), Vector([1, 2])
    en_espanol = [repr(SEL(A, b)), repr(Homogenea(A)), repr(Determinante(I(2)))]
    assert idioma() == 'es'
    assert en_espanol[0] == 'Vector([1, 0]) + Combinaciones lineales de (Sistema([Vector([-2, 1]);]))'
    assert en_espanol[2] == 'Valor del determinante: 1'
    try:
        idioma('en')
        assert idioma() == 'en'
        assert repr(SEL(A, b)) == 'Vector([1, 0]) + Linear combinations of (Sistema([Vector([-2, 1]);]))'
        assert repr(Homogenea(A)) == 'Linear combinations of (Sistema([Vector([-2, 1]);]))'
        assert repr(Determinante(I(2))) == 'Value of the determinant: 1'
        assert SEL(A, b).tex == (idioma('es'), SEL(A, b).tex, idioma('en'))[1]
        with pytest.raises(ValueError):
            idioma('fr')
        assert idioma() == 'en'
    finally:
        idioma('es')
    assert [repr(SEL(A, b)), repr(Homogenea(A)), repr(Determinante(I(2)))] == en_espanol
    assert set(TEXTOS_DE_SALIDA['es']) == set(TEXTOS_DE_SALIDA['en'])


def test_desarrollo_compila_con_mathtools(tmp_path):
    """Con mathtools, \\begin{aligned}& seguido de un array no compila"""
    import shutil, subprocess
    if not shutil.which('pdflatex'):
        pytest.skip('no hay pdflatex')
    a, b, c, d = sympy.symbols('a b c d')
    A = Matrix([[a, b], [c, d]])
    S = SEL(Sistema([A**0, A]), A**2)
    E = ElimGJ(Matrix([[0, 2, 2], [1, -2, -8], [1, 1, 1]]))
    formulas = [desarrollo(S, por_linea=1, compacto=1.5), desarrollo(E, por_linea=2),
                desarrollo(E, por_linea=1, pasos=[2]), desarrollo(E, pasos=[], omitidos='puntos', compacto=True)]
    for paquete in ('mathtools', 'amsmath'):
        fuente = tmp_path / (paquete + '.tex')
        fuente.write_text('\\documentclass{book}\\usepackage{' + paquete + '}\\begin{document}\\footnotesize\n'
                          + '\n'.join('\\[' + f + '\\]' for f in formulas) + '\n\\end{document}\n', encoding='utf-8')
        orden = ['pdflatex', '-interaction=batchmode', '-halt-on-error', fuente.name]
        assert subprocess.run(orden, cwd=tmp_path, capture_output=True).returncode == 0, paquete


# --- fallos corregidos al documentar la inversa y SEL
def test_inversa_de_matriz_singular():
    S = Matrix([[1, 2], [2, 4]])
    for inversa in (InvMat, InvMatF, InvMatFC, lambda M: M.inversa()):
        with pytest.raises(ValueError, match='singular'):
            inversa(S)
    A = Matrix([[5, 0], [2, -2]])
    assert InvMat(A) == InvMatF(A) == InvMatFC(A) == A.inversa()
    assert A*InvMatF(A) == I(2)
    a = sympy.symbols('a')
    P = Matrix([[1, 2], [2, a]])
    with pytest.raises(ValueError, match='singular'):
        InvMatF(P, sust=[(a, 4)])

def test_SEL_sin_solucion():
    N = SEL(Matrix([[1, 2], [2, 4]]), Vector([1, 3]))
    assert repr(N) == 'set()' and N.solP == set() and N.latex() == '\\emptyset'

def test_extDiag_rechaza_lo_que_no_es_BlockM():
    with pytest.raises(ValueError):
        I(2).extDiag([3])
    assert I(1).extDiag(I(1)) == I(2)


# --- regresión de la 0.3.0 detectada con el banco de pruebas del material (examen final de mayo de 2026)
def test_SEL_con_sistemas_simbolicos():
    u, h, p = (VectorSimbol(I(5)|i, n) for i, n in ((1, 'u'), (2, 'h'), (3, 'p')))
    B = Sistema([u, h, p])
    a = sympy.symbols(r'\alpha')
    coordenadas = Matrix([Vector([1, 2, 0]), Vector([0, -1, 1]), Vector([1, a, 0])])
    solucion = SEL(B*coordenadas, B*Vector([2, 1, 2]), sust=[(a, 1)])
    assert solucion.solP == Vector([1, 2, 1])            # como con la versión 0.2.2
    assert (u - u).es_nulo() and (2*u).no_es_nulo()
    assert u.no_es_nulo() and not u.es_nulo() and u.no_es_nulo([(a, 1)])


def test_fracciones_escritas_como_division_de_python():
    """1/3 en Python es un float; se convierte en un tercio, no en su escritura decimal"""
    R = sympy.Rational
    assert Vector([1/3, 2/3, 1/7]).lista == [R(1, 3), R(2, 3), R(1, 7)]
    assert (1/3) * Vector([3, 6]) == Vector([1, 2])
    assert Matrix([[1/3, 1], [1, 3]]).rango() == 1
    assert racional(sympy.Float(1/3) * sympy.Symbol('a')) == sympy.Symbol('a') / 3
    # los decimales exactos siguen igual
    assert Vector([0.1, 0.999, 2.25, 0.75]).lista == [R(1, 10), R(999, 1000), R(9, 4), R(3, 4)]
    assert racional(0.1234567) == R(1234567, 10**7)
    # sin fracción sencilla se lee la escritura decimal, como antes
    assert racional(2**0.5) == R('1.4142135623730951')
    assert racional(0.1 + 0.2) == R('0.30000000000000004')


def test_es_antisimetrica():
    assert Matrix([[0, 2], [-2, 0]]).es_antisimetrica()
    assert not Matrix([[1, 2], [-2, 0]]).es_antisimetrica()      # diagonal no nula
    assert not Matrix([[0, 2], [2, 0]]).es_antisimetrica()       # simétrica
    assert not Matrix([[0, 1, 2], [-1, 0, 3]]).es_antisimetrica()  # no cuadrada
    assert M0(3).es_antisimetrica() and M0(3).es_simetrica()

def test_suma_con_un_numero_a_la_izquierda_da_un_mensaje_en_castellano():
    import pytest
    a, b = Vector([1, 2, 3]), Vector([4, 5, 6])
    for expresion in (lambda: 3 + a, lambda: 3 - a, lambda: a|3 + b|3, lambda: 3 + Matrix([[1]])):
        with pytest.raises(ValueError, match='Solo se (suman|restan) Sistemas'):
            expresion()
    with pytest.raises(ValueError, match='use paréntesis'):
        a|3 + b|3
    assert (a|3) + (b|3) == 9
    # lo que ya funcionaba sigue igual
    assert a + b == Vector([5, 7, 9]) and a - b == Vector([-3, -3, -3])
    assert sum([a, b], V0(3)) == Vector([5, 7, 9])
    x = sympy.Symbol('x')
    assert x * a == Vector([x, 2*x, 3*x])


def test_error_de_producto_con_numeros_y_traducido():
    """El mensaje dice qué se multiplica y los números que no coinciden"""
    A = Matrix([[1, 2, 3], [4, 5, 6]])
    casos = [
        (lambda: Vector([1, 2, 3]) * Vector([1, 2]),
         'No se pueden multiplicar: el vector de la izquierda tiene 3 componentes y el vector de la derecha tiene 2 componentes',
         'Cannot multiply: the vector on the left has 3 components and the vector on the right has 2 components'),
        (lambda: A * Vector([1, 2]),
         'No se pueden multiplicar: la matriz de la izquierda tiene 3 columnas y el vector de la derecha tiene 2 componentes',
         'Cannot multiply: the matrix on the left has 3 columns and the vector on the right has 2 components'),
        (lambda: A * A,
         'No se pueden multiplicar: la matriz de la izquierda tiene 3 columnas y la matriz de la derecha tiene 2 filas',
         'Cannot multiply: the matrix on the left has 3 columns and the matrix on the right has 2 rows'),
        (lambda: Vector([1]) * A,
         'No se pueden multiplicar: el vector de la izquierda tiene 1 componente y la matriz de la derecha tiene 2 filas',
         'Cannot multiply: the vector on the left has 1 component and the matrix on the right has 2 rows'),
        (lambda: Sistema([A, A, A]) * Vector([1, 2]),
         'No se pueden multiplicar: el sistema de la izquierda tiene 3 elementos y el vector de la derecha tiene 2 componentes',
         'Cannot multiply: the system on the left has 3 elements and the vector on the right has 2 components'),
    ]
    try:
        for codigo, columna in (('es', 1), ('en', 2)):
            idioma(codigo)
            for caso in casos:
                with pytest.raises(ValueError) as error:
                    caso[0]()
                assert str(error.value) == caso[columna]
    finally:
        idioma('es')
    assert Vector([1, 2, 3]) * Vector([1, 1, 1]) == 6
    assert A * Vector([1, 1, 1]) == Vector([6, 15])


def test_simplify_factor_expand_de_una_T():
    """Solo cambian los coeficientes: los intercambios siguen siendo conjuntos"""
    A = Matrix([[1, 2], [3, 4]])
    for funcion in (simplify, factor, expand):
        t = funcion(T([{1, 2}, (a*(a + 1) - a, 2), (2, 1, 2)]))
        assert isinstance(t, T)
        assert isinstance(t.abreviaturas[0], set)
        assert [type(indice) for indice in t.abreviaturas[2][1:]] == [int, int]
        # un intercambio simplificado sigue intercambiando
        assert Matrix(A) & funcion(T({1, 2})) == Matrix([[2, 1], [4, 3]])
    assert simplify(T((a*(a + 1) - a, 2))).abreviaturas == [(a**2, 2)]
    assert factor(T((a**2 - 1, 2))).abreviaturas == [((a - 1)*(a + 1), 2)]
    assert expand(T(((a - 1)*(a + 1), 2, 1))).abreviaturas == [(a**2 - 1, 2, 1)]
    # como métodos hacen lo mismo que las funciones, y devuelven la T
    assert T((a*(a + 1) - a, 2)).simplify().abreviaturas == [(a**2, 2)]
    assert T((a**2 - 1, 2)).factor().abreviaturas == [((a - 1)*(a + 1), 2)]
    assert T(((a - 1)*(a + 1), 2, 1)).expand().abreviaturas == [(a**2 - 1, 2, 1)]
    t = T((a**2 - 1, 2))
    assert t.factor() is t


def test_metodos_simplify_factor_expand_de_un_Sistema_devuelven_el_Sistema():
    """A.factor() modifica A y lo devuelve, igual que factor(A)"""
    A = Matrix([[a**2 - 1, 1], [0, a*(a + 1) - a]])
    assert A.factor() is A
    assert (1|A|1) == (a - 1)*(a + 1)
    assert A.expand() is A and (1|A|1) == a**2 - 1
    assert A.simplify() is A
    v = Vector([a*(a + 1) - a, 1])
    assert v.simplify() is v and (v|1) == a**2
