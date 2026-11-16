#!/usr/bin/env python3
"""Banco de pruebas con el código Python del material real (libro, clases y exámenes).

PythonTeX guarda en un fichero `.pytxcode` todo el código Python de cada
documento que se compila. Este programa vuelve a ejecutar esos ficheros con
una versión concreta de NAcAL y guarda lo que PythonTeX insertaría en el
documento. Comparando dos ejecuciones se sabe qué documentos cambian de
salida (o dejan de funcionar) al cambiar de versión, sin compilar LaTeX.

No modifica nada fuera del directorio de resultados: cada caso se ejecuta en
un directorio temporal con enlaces a los ficheros del original.

Además de los `.pytxcode` ejecuta el código que los alumnos ven y pueden
copiar: los recuadros `pyverbatim` de los apuntes y problemas de clase (que
PythonTeX muestra pero no ejecuta) y, con `--notebooks`, los Notebooks, con
el guion `ejecutar_notebooks.py` de su repositorio.

Uso:
    python3 tests/banco_material.py ejecutar ETIQUETA [--nacal ORIGEN] [--notebooks]
    python3 tests/banco_material.py comparar ETIQUETA_A ETIQUETA_B
    python3 tests/banco_material.py casos

ORIGEN indica la versión de NAcAL:
    instalada       la que encuentra Python (por defecto)
    trabajo         la copia de trabajo de este repositorio
    git:REVISION    la de una revisión de git (rama, etiqueta o commit)
    RUTA            un directorio que contiene el paquete nacal/

Los `.pytxcode` se buscan en los directorios de la variable de entorno
NACAL_BANCO_FUENTES (separados por ':'); por defecto, en las compilaciones de
~/temporal/nacal-banco/trabajo y en el repositorio del libro. Los resultados
se guardan en NACAL_BANCO_SALIDAS (por defecto ~/temporal/nacal-banco-python).

Los recuadros se buscan en los `.tex` de NACAL_BANCO_RECUADROS (por defecto
../MatematicasII-SRC/clases) y los Notebooks se ejecutan con el guion de
NACAL_BANCO_NOTEBOOKS (por defecto ../NAcAL-Jupyter-Notebooks-SRC). Ambos se
leen del repositorio tal como esté en ese momento: para comparar dos versiones
de NAcAL hay que ejecutar las dos seguidas.
"""
import argparse
import concurrent.futures
import difflib
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent.parent
FUENTES = [pathlib.Path(p).expanduser() for p in os.environ.get(
    'NACAL_BANCO_FUENTES',
    '~/temporal/nacal-banco/trabajo:' + str(REPO.parent / 'CursoDeAlgebraLineal-SRC')).split(':')]
SALIDAS = pathlib.Path(os.environ.get('NACAL_BANCO_SALIDAS', '~/temporal/nacal-banco-python')).expanduser()

RECUADROS = pathlib.Path(os.environ.get('NACAL_BANCO_RECUADROS',
                                       str(REPO.parent / 'MatematicasII-SRC' / 'clases'))).expanduser()
NOTEBOOKS = pathlib.Path(os.environ.get('NACAL_BANCO_NOTEBOOKS',
                                       str(REPO.parent / 'NAcAL-Jupyter-Notebooks-SRC'))).expanduser()

CABECERA = re.compile(r'^=>PYTHONTEX#', re.M)
RECUADRO = re.compile(r'\\begin\{pyverbatim\}[^\n]*\n(.*?)^[ \t]*\\end\{pyverbatim\}', re.M | re.S)

# Programa que ejecuta, uno tras otro y en el mismo espacio de nombres, los
# recuadros de un fichero como lo haría un alumno en un terminal de Python:
# escribe lo que se imprime, el valor de cada expresión y los errores.
EJECUTOR_DE_RECUADROS = r"""
import ast, contextlib, io, json, sys, textwrap
recuadros = json.load(open(sys.argv[1], encoding='utf-8'))
entorno = {}
exec('from nacal import *', entorno)
def muestra(valor):
    if valor is None:
        return
    print(repr(valor))
    if hasattr(valor, 'latex'):
        print('    latex:', valor.latex())
sys.displayhook = muestra
for k, codigo in enumerate(recuadros, 1):
    print('#### recuadro', k)
    salida = io.StringIO()
    try:
        with contextlib.redirect_stdout(salida):
            for orden in ast.parse(textwrap.dedent(codigo)).body:
                exec(compile(ast.Interactive([orden]), '<recuadro>', 'single'), entorno)
        print(salida.getvalue(), end='')
    except Exception as error:
        print(salida.getvalue(), end='')
        print('ERROR', type(error).__name__ + ':', error)
"""


def casos():
    """Diccionario {nombre: fichero .pytxcode}, sin repetidos ni ficheros sin código"""
    vistos, encontrados = set(), {}
    for fuente in FUENTES:
        if not fuente.is_dir():
            continue
        candidatos = sorted(fuente.glob('*.pytxcode')) + sorted(fuente.glob('*/*.pytxcode'))
        for fichero in candidatos:
            texto = fichero.read_text(encoding='utf-8', errors='replace')
            if len(CABECERA.findall(texto)) < 2 or 'nacal' not in texto:
                continue                                  # no ejecuta código que use NAcAL
            huella = hashlib.sha1(texto.encode()).hexdigest()
            if huella in vistos:
                continue                                  # mismo código que otro caso
            vistos.add(huella)
            padre = fichero.parent.name if fichero.parent != fuente else fuente.name
            encontrados[padre + '--' + fichero.stem] = fichero
    return encontrados


def directorio_de_nacal(origen):
    """Directorio que se antepone a PYTHONPATH (None para la versión instalada)"""
    if origen == 'instalada':
        return None
    if origen == 'trabajo':
        return REPO
    if origen.startswith('git:'):
        revision = origen[4:]
        commit = subprocess.run(['git', '-C', str(REPO), 'rev-parse', '--short', revision],
                                capture_output=True, text=True, check=True).stdout.strip()
        destino = SALIDAS / 'versiones' / commit
        if not (destino / 'nacal' / 'nacal.py').exists():
            destino.mkdir(parents=True, exist_ok=True)
            archivo = subprocess.run(['git', '-C', str(REPO), 'archive', commit, 'nacal'],
                                     capture_output=True, check=True).stdout
            subprocess.run(['tar', '-x', '-C', str(destino)], input=archivo, check=True)
        return destino
    ruta = pathlib.Path(origen).expanduser().resolve()
    if not (ruta / 'nacal' / '__init__.py').exists():
        sys.exit('No hay un paquete nacal en ' + str(ruta))
    return ruta


def normaliza(texto, temporal):
    """Quita de una salida lo que cambia de una ejecución a otra"""
    texto = re.sub(r'^%Last time of file creation:.*\n', '', texto, flags=re.M)
    texto = texto.replace(str(temporal), '<dir>')
    texto = re.sub(r'/[^\s"\']*/nacal/(nacal|__init__)\.py', r'<nacal>/\1.py', texto)
    texto = re.sub(r'0x[0-9a-f]{6,}', '0x...', texto)          # direcciones de memoria
    texto = re.sub(r', line \d+, in ', ', line N, in ', texto)  # las líneas cambian con la versión
    return texto


def ejecuta_caso(nombre, fichero, nacal, destino):
    """Ejecuta PythonTeX sobre una copia del caso y guarda las salidas normalizadas"""
    trabajo = tempfile.mkdtemp(prefix='banco-')
    try:
        for original in fichero.parent.iterdir():
            if original.name.startswith('pythontex-files-') or original.suffix == '.pytxcode':
                continue
            os.symlink(original, os.path.join(trabajo, original.name))
        shutil.copy(fichero, trabajo)
        entorno = dict(os.environ)
        if nacal is not None:
            entorno['PYTHONPATH'] = str(nacal) + os.pathsep + entorno.get('PYTHONPATH', '')
        try:
            proceso = subprocess.run(['pythontex', '--interpreter', 'python:python3', '--rerun', 'always',
                                      fichero.stem], cwd=trabajo, env=entorno,
                                     capture_output=True, text=True, timeout=1800)
            mensajes = proceso.stdout + proceso.stderr
        except subprocess.TimeoutExpired:
            mensajes = 'TIEMPO AGOTADO\n'
        salida = pathlib.Path(trabajo) / ('pythontex-files-' + fichero.stem)
        partes = []
        if salida.is_dir():
            for f in sorted(salida.iterdir()):
                if f.suffix in ('.pytxmcr', '.stdout', '.tex', '.txt'):
                    partes.append('#### ' + f.name + '\n' + f.read_text(encoding='utf-8', errors='replace'))
        (destino / (nombre + '.salida')).write_text(normaliza('\n'.join(partes), trabajo), encoding='utf-8')
        (destino / (nombre + '.mensajes')).write_text(normaliza(mensajes, trabajo), encoding='utf-8')
        resumen = re.search(r'(\d+) error\(s\), (\d+) warning\(s\)', mensajes)
        return nombre, (int(resumen.group(1)) if resumen else -1)
    finally:
        shutil.rmtree(trabajo, ignore_errors=True)


def recuadros():
    """Diccionario {nombre: lista de recuadros pyverbatim} de los ficheros de clase"""
    encontrados = {}
    if RECUADROS.is_dir():
        for fichero in sorted(RECUADROS.rglob('*.tex')):
            codigo = RECUADRO.findall(fichero.read_text(encoding='utf-8', errors='replace'))
            if codigo:
                nombre = 'recuadros--' + str(fichero.relative_to(RECUADROS).with_suffix('')).replace(os.sep, '__')
                encontrados[nombre] = codigo
    return encontrados


def ejecuta_recuadros(nombre, codigo, nacal, destino):
    """Ejecuta los recuadros de un fichero y guarda lo que vería el alumno"""
    trabajo = tempfile.mkdtemp(prefix='banco-')
    try:
        (pathlib.Path(trabajo) / 'recuadros.json').write_text(json.dumps(codigo), encoding='utf-8')
        (pathlib.Path(trabajo) / 'ejecutor.py').write_text(EJECUTOR_DE_RECUADROS, encoding='utf-8')
        entorno = dict(os.environ)
        if nacal is not None:
            entorno['PYTHONPATH'] = str(nacal) + os.pathsep + entorno.get('PYTHONPATH', '')
        try:
            proceso = subprocess.run([sys.executable, 'ejecutor.py', 'recuadros.json'], cwd=trabajo, env=entorno,
                                     capture_output=True, text=True, timeout=600)
            salida, mensajes = proceso.stdout, proceso.stderr
        except subprocess.TimeoutExpired:
            salida, mensajes = '', 'TIEMPO AGOTADO\n'
        (destino / (nombre + '.salida')).write_text(normaliza(salida, trabajo), encoding='utf-8')
        (destino / (nombre + '.mensajes')).write_text(normaliza(mensajes, trabajo), encoding='utf-8')
        return nombre, salida.count('\nERROR ') + (1 if mensajes.strip() else 0)
    finally:
        shutil.rmtree(trabajo, ignore_errors=True)


def ejecuta_notebooks(nacal, destino):
    """Ejecuta los Notebooks con el guion de su repositorio y guarda las salidas de cada celda"""
    guion = NOTEBOOKS / 'ejecutar_notebooks.py'
    if not guion.exists():
        print('Notebooks: no se encuentra', guion)
        return
    trabajo = tempfile.mkdtemp(prefix='banco-')
    try:
        entorno = dict(os.environ)
        if nacal is not None:
            entorno['PYTHONPATH'] = str(nacal) + os.pathsep + entorno.get('PYTHONPATH', '')
        proceso = subprocess.run([sys.executable, str(guion), 'ejecutar', trabajo], env=entorno,
                                 capture_output=True, text=True, timeout=3600)
        ejecutados = sorted(pathlib.Path(trabajo).rglob('*.json'))
        for fichero in ejecutados:
            datos = json.loads(fichero.read_text(encoding='utf-8'))
            celdas = datos['celdas'] if isinstance(datos, dict) else datos
            celdas = [c for c in celdas if c.get('celda', 0) >= 0]     # sin la celda que anota la versión
            nombre = 'notebooks--' + str(fichero.relative_to(trabajo).with_suffix('')).replace(os.sep, '__')
            texto = json.dumps(celdas, ensure_ascii=False, indent=1, sort_keys=True)
            (destino / (nombre + '.salida')).write_text(normaliza(texto, trabajo), encoding='utf-8')
            (destino / (nombre + '.mensajes')).write_text('', encoding='utf-8')
        print('Notebooks ejecutados:', len(ejecutados))
        if proceso.returncode or not ejecutados:
            print('   el guion de los Notebooks terminó con código', proceso.returncode)
            print('   ' + proceso.stderr.strip()[-600:])
    finally:
        shutil.rmtree(trabajo, ignore_errors=True)


def ejecutar(etiqueta, origen, procesos, con_notebooks=False):
    nacal = directorio_de_nacal(origen)
    destino = SALIDAS / 'resultados' / etiqueta
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    todos = casos()
    de_clase = recuadros()
    print('NAcAL:', origen, '->', nacal or '(instalada)')
    print('casos:', len(todos))
    print('ficheros con recuadros de código:', len(de_clase), '(%d recuadros)' % sum(map(len, de_clase.values())))
    con_errores = {}
    with concurrent.futures.ThreadPoolExecutor(procesos) as grupo:
        tareas = [grupo.submit(ejecuta_caso, n, f, nacal, destino) for n, f in todos.items()]
        tareas += [grupo.submit(ejecuta_recuadros, n, c, nacal, destino) for n, c in de_clase.items()]
        for tarea in concurrent.futures.as_completed(tareas):
            nombre, errores = tarea.result()
            if errores:
                con_errores[nombre] = errores
    (destino / 'ORIGEN').write_text(origen + '\n' + str(nacal or 'instalada') + '\n')
    print('casos con errores de ejecución:', len(con_errores))
    for nombre in sorted(con_errores):
        print('   ', nombre, '(sin resumen de PythonTeX)' if con_errores[nombre] < 0 else con_errores[nombre])
    if con_notebooks:
        ejecuta_notebooks(nacal, destino)
    print('resultados en', destino)


def comparar(a, b):
    da, db = SALIDAS / 'resultados' / a, SALIDAS / 'resultados' / b
    for d in (da, db):
        if not d.is_dir():
            sys.exit('No existe la ejecución ' + d.name)
    nombres_a = {f.stem for f in da.glob('*.salida')}
    nombres_b = {f.stem for f in db.glob('*.salida')}
    informe = SALIDAS / 'informes' / (a + '--' + b)
    if informe.exists():
        shutil.rmtree(informe)
    informe.mkdir(parents=True)
    iguales, distintos = 0, []
    for nombre in sorted(nombres_a & nombres_b):
        diferencias = []
        for sufijo in ('.salida', '.mensajes'):
            ta = (da / (nombre + sufijo)).read_text(encoding='utf-8')
            tb = (db / (nombre + sufijo)).read_text(encoding='utf-8')
            if ta != tb:
                diferencias += list(difflib.unified_diff(ta.splitlines(), tb.splitlines(),
                                                         a + '/' + nombre + sufijo, b + '/' + nombre + sufijo,
                                                         lineterm='', n=1))
        if diferencias:
            distintos.append(nombre)
            (informe / (nombre + '.diff')).write_text('\n'.join(diferencias) + '\n', encoding='utf-8')
        else:
            iguales += 1
    print('%s frente a %s: iguales %d, distintos %d, solo en %s %d, solo en %s %d'
          % (a, b, iguales, len(distintos), a, len(nombres_a - nombres_b), b, len(nombres_b - nombres_a)))
    for nombre in distintos:
        print('    distinto:', nombre)
    if distintos:
        print('diferencias en', informe)
    return 1 if distintos or nombres_a != nombres_b else 0


def main():
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ordenes = analizador.add_subparsers(dest='orden', required=True)
    e = ordenes.add_parser('ejecutar')
    e.add_argument('etiqueta')
    e.add_argument('--nacal', default='instalada')
    e.add_argument('--procesos', type=int, default=max(1, (os.cpu_count() or 2) // 2))
    e.add_argument('--notebooks', action='store_true', help='ejecuta también los Notebooks (unos minutos)')
    c = ordenes.add_parser('comparar')
    c.add_argument('a')
    c.add_argument('b')
    ordenes.add_parser('casos')
    args = analizador.parse_args()
    if args.orden == 'ejecutar':
        ejecutar(args.etiqueta, args.nacal, args.procesos, args.notebooks)
    elif args.orden == 'comparar':
        sys.exit(comparar(args.a, args.b))
    else:
        todos = casos()
        for nombre, fichero in todos.items():
            print(nombre, '<-', fichero)
        de_clase = recuadros()
        for nombre, codigo in de_clase.items():
            print(nombre, '<-', len(codigo), 'recuadros')
        print(len(todos), 'casos y', len(de_clase), 'ficheros con recuadros')


if __name__ == '__main__':
    main()
