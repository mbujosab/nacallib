#!/usr/bin/env python3
"""Vuelve a ejecutar los ejemplos de nacal_src.org y pone al día sus resultados

Ejecuta, en el orden del documento, todos los bloques `jupyter-python` que se
exportan (con el módulo de este repositorio, no con el instalado) y reescribe
solo los `#+RESULTS:` que no coinciden con la salida real. No hace falta
Emacs ni Jupyter.

Uso, desde cualquier carpeta:

    python3 herramientas/actualiza_resultados.py            # reescribe
    python3 herramientas/actualiza_resultados.py --mirar    # solo dice cuáles cambiarían

Antes hay que generar el módulo (tangle) si se ha cambiado su código, y
guardar el fichero en Emacs. Después, en Emacs, `M-x revert-buffer`.
Usa la misma maquinaria que tests/test_documentacion.py.
"""
import contextlib
import io
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / 'tests'))
sys.path.insert(0, str(RAIZ))

import test_documentacion as pruebas
import nacal.nacal as modulo


def muestra(objeto, *args, **kwargs):
    # lo que Jupyter guarda como texto de un Math(...)
    datos = getattr(objeto, 'data', None)
    print('$\\displaystyle ' + datos + '$' if isinstance(datos, str) else repr(objeto))


def main():
    solo_mirar = '--mirar' in sys.argv[1:]
    if not solo_mirar and (RAIZ / '.#nacal_src.org').is_symlink():
        sys.exit('nacal_src.org está abierto en Emacs con cambios sin guardar: guárdelo antes')
    texto = pruebas.ORG.read_text(encoding='utf-8')
    entorno = {}
    exec(pruebas.PREAMBULO, entorno)
    entorno['display'] = muestra
    modulo.display = muestra
    cambios = []
    for bloque in pruebas.BLOQUE_CON_RESULTADO.finditer(texto):
        if pruebas.no_se_exporta(texto, bloque.start()):
            continue
        codigo, resultado = bloque.groups()
        salida = io.StringIO()
        with contextlib.redirect_stdout(salida):
            pruebas.ejecuta_como_jupyter(codigo, entorno)
        if resultado is None:
            continue
        real, esperado = salida.getvalue().strip(), resultado.strip()
        if real != esperado and real != esperado.replace('\\n', '\n'):
            linea = texto.count('\n', 0, bloque.start()) + 1
            print('línea %d: %s' % (linea, codigo.strip().split('\n')[-1]))
            if esperado.startswith(':RESULTS:'):
                sys.exit('  ese resultado está en un cajón :RESULTS: y hay que corregirlo a mano')
            cambios.append((bloque.start(2), bloque.end(2), real))
    if not solo_mirar:
        for inicio, fin, real in reversed(cambios):
            texto = texto[:inicio] + real + texto[fin:]
        pruebas.ORG.write_text(texto, encoding='utf-8')
    print('%d resultados %s' % (len(cambios), 'cambiarían' if solo_mirar else 'actualizados'))


if __name__ == '__main__':
    main()
