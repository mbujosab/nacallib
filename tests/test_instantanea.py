"""Pruebas de regresión ("instantánea") del módulo nacal.

Cada caso evalúa una expresión y guarda su repr y su LaTeX en
instantanea.json.  La prueba falla si el resultado actual difiere del
guardado.  Si el cambio es intencionado, regenere la instantánea con

    NACAL_ACTUALIZA=1 python3 -m pytest tests/test_instantanea.py

y revise el diff de instantanea.json antes de confirmarlo.
"""
import json, os
import pytest
import sympy
from nacal import *

FICHERO = os.path.join(os.path.dirname(__file__), 'instantanea.json')
a, b, x = sympy.symbols('a b x')

A  = lambda: Matrix([[1, 2, 0], [2, 4, 1], [0, 1, 3]])
B  = lambda: Matrix([[1, 2, 3], [2, 4, 6], [1, 0, 1]])           # singular
C  = lambda: Matrix([[1, 1, 2, 0], [2, 2, 1, 1], [0, 0, 3, -1]]) # 3x4
S  = lambda: Matrix([[2, 1, 0], [1, 2, 0], [0, 0, 3]])           # simétrica
P  = lambda: Matrix([[a, 1], [1, a]])                            # con parámetro
v  = lambda: Vector([1, 2, 3])
w  = lambda: Vector([2, 0, -1])

def tex(o):
    return o.tex if hasattr(o, 'tex') else latex(o)

CASOS = {
    # --- álgebra básica y selectores
    'suma_vectores':      lambda: v() + w(),
    'resta_matrices':     lambda: A() - B(),
    'escalar_izda':       lambda: 3 * A(),
    'escalar_dcha':       lambda: A() * sympy.Rational(1, 2),
    'producto_punto':     lambda: v() * w(),
    'matriz_por_vector':  lambda: A() * v(),
    'vector_por_matriz':  lambda: v() * A(),
    'matriz_por_matriz':  lambda: A() * B(),
    'transpuesta':        lambda: ~C(),
    'columna':            lambda: C() | 2,
    'fila':               lambda: 2 | C(),
    'elemento':           lambda: 2 | C() | 3,
    'submatriz':          lambda: (1, 3) | C() | [2, 4],
    'slice':              lambda: Vector([1, 2, 3, 4, 5]) | slice(2, 4),
    'potencia':           lambda: A() ** 3,
    'potencia_negativa':  lambda: A() ** -1,
    'concatena':          lambda: A().concatena(B(), 1),
    'apila':              lambda: A().apila(B(), 1),
    'extDiag':            lambda: A().extDiag(I(2), 1),
    'diag':               lambda: A().diag(),
    'traza':              lambda: A().traza(),
    'norma':              lambda: v().norma(),
    'normalizado':        lambda: v().normalizado(),
    'GS':                 lambda: A().GS(),
    'M0_M1_I':            lambda: Sistema([M0(2, 3), M1(2), I(3), V0(2), V1(2)]),
    'BlockM':             lambda: BlockM([A(), B()]),
    'BlockV_h':           lambda: BlockV([A(), v()], rpr='h'),
    'str_matriz':         lambda: str(C().csis({2}).cele({1})),
    'repr_sistema':       lambda: repr(Sistema([1, v(), A()])),
    # --- transformaciones elementales
    'T_dcha':             lambda: A() & T([(2, 1, 3), {1, 2}, (5, 3)]),
    'T_izda':             lambda: T([(2, 1, 3), {1, 2}, (5, 3)]) & A(),
    'T_composicion':      lambda: T({1, 2}) & T([(2, 4), (2, 1, 3)]),
    'T_inversa':          lambda: T([(2, 4), (2, 1, 3), {1, 2}]) ** -1,
    'T_transpuesta':      lambda: ~T([(2, 4), (2, 1, 3), {1, 2}]),
    'T_espejo':           lambda: T([(2, 4), (2, 1, 3), {1, 2}]).espejo(),
    'T_latex_h':          lambda: T([(2, 4), (2, 1, 3), {1, 2}], rpr='h'),
    # --- eliminación
    'Elim_A':             lambda: Elim(A()),
    'Elim_C':             lambda: Elim(C()),
    'ElimG_C':            lambda: ElimG(C()),
    'ElimGJ_C':           lambda: ElimGJ(C()),
    'ElimF_C':            lambda: ElimF(C()),
    'ElimGF_C':           lambda: ElimGF(C()),
    'ElimGJF_C':          lambda: ElimGJF(C()),
    'ElimGJ_B':           lambda: ElimGJ(B()),
    'elim_1':             lambda: C().elim(1),
    'elim_100':           lambda: C().elim(100),
    'elim_120':           lambda: C().elim(120),
    'Elim_pasos':         lambda: repr(Elim(C()).pasos),
    'ElimGJF_pasos':      lambda: repr(ElimGJF(C()).pasos),
    'Elim_parametro':     lambda: Elim(P()),
    'Elim_sust':          lambda: Elim(P(), 0, [(a, 1)]),
    'Elim_repsust':       lambda: Elim(P(), 0, [(a, 1)], 1),
    'rango':              lambda: [M().rango() for M in (A, B, C, S)],
    'rprElim':            lambda: rprElim(C(), Elim(C()).pasos),
    # --- inversa y determinante
    'InvMat':             lambda: InvMat(A()),
    'InvMatF':            lambda: InvMatF(A()),
    'InvMatFC':           lambda: InvMatFC(A()),
    'inversa':            lambda: A().inversa(),
    'InvMat_parametro':   lambda: InvMat(P()),
    'Determinante_A':     lambda: Determinante(A()),
    'Determinante_B':     lambda: Determinante(B()),
    'Determinante_tex':   lambda: Determinante(A()).tex,
    'Determinante_P':     lambda: Determinante(P()).tex,
    'det_Laplace':        lambda: [A().det(), B().det(), P().det()],
    'determinante':       lambda: [A().determinante(), B().determinante(), P().determinante()],
    # --- sistemas de ecuaciones
    'Homogenea_B':        lambda: Homogenea(B()),
    'Homogenea_B_tex':    lambda: Homogenea(B()).tex,
    'Homogenea_A':        lambda: Homogenea(A()),
    'Homogenea_C':        lambda: Homogenea(C()),
    'SEL_det':            lambda: SEL(A(), v()),
    'SEL_det_tex':        lambda: SEL(A(), v()).tex,
    'SEL_indet':          lambda: SEL(B(), Vector([1, 2, 1])),
    'SEL_indet_tex':      lambda: SEL(B(), Vector([1, 2, 1])).tex,
    'SEL_incompatible':   lambda: SEL(B(), Vector([1, 0, 1])),
    'SEL_C':              lambda: SEL(C(), v()),
    'SEL_matrices':       lambda: SEL(Sistema([Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]]), Matrix([[1, 1], [1, 0]]), Matrix([[0, 0], [0, 1]])]), Matrix([[2, 3], [3, 5]])),
    'SEL_matrices_tex':   lambda: SEL(Sistema([Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]]), Matrix([[1, 1], [1, 0]]), Matrix([[0, 0], [0, 1]])]), Matrix([[2, 3], [3, 5]])).tex,
    'SEL_matrices_incomp':lambda: SEL(Sistema([Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]])]), Matrix([[2, 3], [4, 5]])),
    'Homogenea_matrices': lambda: Homogenea(Sistema([Matrix([[1, 0], [0, 0]]), Matrix([[0, 1], [1, 0]]), Matrix([[1, 1], [1, 0]])])),
    'SEL_solP':           lambda: SEL(C(), v()).solP,
    'sel_metodo':         lambda: C().sel(v()),
    # --- subespacios y espacios afines
    'SubEspacio_sgen':    lambda: SubEspacio(Sistema([v(), w(), v() + w()])),
    'SubEspacio_matriz':  lambda: SubEspacio(B()),
    'SubEspacio_dim':     lambda: [SubEspacio(Sistema([v(), w(), v() + w()])).dim, SubEspacio(B()).dim],
    'SubEspacio_cart':    lambda: SubEspacio(Sistema([v(), w()])).cart,
    'SubEspacio_suma':    lambda: SubEspacio(Sistema([v()])) + SubEspacio(Sistema([w()])),
    'SubEspacio_inter':   lambda: SubEspacio(Sistema([v(), w()])) & SubEspacio(Sistema([v(), Vector([0, 0, 1])])),
    'SubEspacio_ortog':   lambda: ~SubEspacio(Sistema([v(), w()])),
    'SubEspacio_igual':   lambda: SubEspacio(Sistema([v(), w()])) == SubEspacio(Sistema([v() + w(), w()])),
    'SubEspacio_in':      lambda: [v() in SubEspacio(Sistema([v(), w()])), Vector([0, 0, 1]) in SubEspacio(Sistema([v(), w()]))],
    'SubEspacio_matrices':lambda: SubEspacio(Sistema([I(2), M1(2)])),
    'span':               lambda: C().span(),
    'espacio_nulo':       lambda: C().espacio_nulo(),
    'EAfin':              lambda: EAfin(Sistema([v(), w()]), Vector([1, 1, 1])),
    'EAfin_cart':         lambda: EAfin(Sistema([v(), w()]), Vector([1, 1, 1])).EcCartesianas(),
    'EAfin_inter':        lambda: EAfin(Sistema([v(), w()]), Vector([1, 1, 1])) & EAfin(Sistema([v(), Vector([0, 0, 1])]), Vector([0, 1, 0])),
    'EAfin_in':           lambda: Vector([2, 3, 4]) in EAfin(Sistema([v(), w()]), Vector([1, 1, 1])),
    # --- diagonalización
    'DiagonalizaS':       lambda: DiagonalizaS(S(), [1, 3, 3]),
    'DiagonalizaS_tex':   lambda: DiagonalizaS(S(), [1, 3, 3]).tex,
    'DiagonalizaS_S':     lambda: DiagonalizaS(S(), [1, 3, 3]).S,
    'DiagonalizaO':       lambda: DiagonalizaO(S(), [1, 3, 3]),
    'DiagonalizaO_Q':     lambda: DiagonalizaO(S(), [1, 3, 3]).Q,
    'DiagonalizaC':       lambda: DiagonalizaC(S()),
    'DiagonalizaC_tex':   lambda: DiagonalizaC(S()).tex,
    'DiagonalizaC_B':     lambda: DiagonalizaC(S()).B,
    'DiagonalizaC_ceros': lambda: DiagonalizaC(Matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])).tex,
    'DiagonalizaC_var1':  lambda: DiagonalizaC(S(), variante=1),
    # --- simbólicos y funciones lineales
    'VectorSimbol':       lambda: VectorSimbol(v(), 'a'),
    'MatrixSimbol':       lambda: MatrixSimbol(A(), 'a'),
    'subs':               lambda: P().subs([(a, 2)]),
    'FuncionLineal':      lambda: FuncionLineal([Vector([1, 0]), Vector([0, 1])], [Vector([1, 1, 0]), Vector([0, 1, 1])]),
    'FL_imagen_de':       lambda: FuncionLineal([Vector([1, 0]), Vector([0, 1])], [Vector([1, 1, 0]), Vector([0, 1, 1])]).imagen_de(Vector([2, 3])),
    'FL_nucleo':          lambda: FuncionLineal([Vector([1, 0]), Vector([0, 1])], [Vector([1, 1]), Vector([2, 2])]).nucleo,
}

def evalua(nombre):
    try:
        o = CASOS[nombre]()
    except Exception as e:      # los errores también forman parte del comportamiento
        o = 'EXCEPCION ' + type(e).__name__ + ': ' + str(e)
    if isinstance(o, str):
        return {'repr': o, 'tex': o}
    sin_repr = type(o).__repr__ is object.__repr__   # SubEspacio, EAfin, ...
    return {'repr': type(o).__name__ if sin_repr else repr(o), 'tex': tex(o)}

def carga():
    with open(FICHERO, encoding='utf8') as f:
        return json.load(f)

if os.environ.get('NACAL_ACTUALIZA'):
    with open(FICHERO, 'w', encoding='utf8') as f:
        json.dump({n: evalua(n) for n in CASOS}, f, indent=1, ensure_ascii=False, sort_keys=True)

@pytest.mark.parametrize('nombre', sorted(CASOS))
def test_instantanea(nombre):
    assert evalua(nombre) == carga()[nombre]
