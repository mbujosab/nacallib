"""Nombres que exporta `from nacal import *`."""
import types

import nacal
import nacal.nacal as modulo

# nombres públicos del módulo que deliberadamente NO se exportan
INTERNOS = {
    'tempfile', 'NumberTypes', 'RistraTypes', 'CreaLista', 'es_ristra',
    'es_ristra_de_numeros', 'es_funcion_racional', 'elementoPivote',
    'elementoAntiPivote', 'fin_del_argumento', 'trozos_del_desarrollo',
    'flecha_resumen', 'TEXTOS_DE_SALIDA', 'idioma_en_uso', 'texto_de_salida',
    'TexDesarrollo', 'texYpasos',
    'fracciones_en_uso', 'salto_de_fila',
    'AVISO_DE_PRECEDENCIA',
}


def test_todo_lo_exportado_existe():
    assert len(set(modulo.__all__)) == len(modulo.__all__)
    for nombre in modulo.__all__:
        assert hasattr(modulo, nombre), nombre
        assert hasattr(nacal, nombre), nombre

def test_cada_nombre_publico_esta_clasificado():
    """Al añadir algo al módulo hay que decidir si se exporta o es interno"""
    publicos = {n for n in dir(modulo) if not n.startswith('_')}
    assert publicos == set(modulo.__all__) | INTERNOS
    assert not set(modulo.__all__) & INTERNOS

def test_import_estrella():
    entorno = {}
    exec('from nacal import *', entorno)
    assert set(entorno) - {'__builtins__'} == set(modulo.__all__)
    assert isinstance(entorno['sympy'], types.ModuleType)
    assert 'tempfile' not in entorno and 'trozos_del_desarrollo' not in entorno
