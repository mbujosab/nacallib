#!/usr/bin/env python3
"""Banco de pruebas con el código Python del material real (libro, clases y exámenes).

PythonTeX guarda en un fichero `.pytxcode` todo el código Python de cada
documento que se compila. Este programa vuelve a ejecutar esos ficheros con
una versión concreta de NAcAL y guarda lo que PythonTeX insertaría en el
documento. Comparando dos ejecuciones se sabe qué documentos cambian de
salida (o dejan de funcionar) al cambiar de versión, sin compilar LaTeX.

No modifica nada fuera del directorio de resultados: cada caso se ejecuta en
un directorio temporal con enlaces a los ficheros del original.

Uso:
    python3 tests/banco_material.py ejecutar ETIQUETA [--nacal ORIGEN]
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
"""
import argparse
import concurrent.futures
import difflib
import hashlib
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

CABECERA = re.compile(r'^=>PYTHONTEX#', re.M)


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


def ejecutar(etiqueta, origen, procesos):
    nacal = directorio_de_nacal(origen)
    destino = SALIDAS / 'resultados' / etiqueta
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)
    todos = casos()
    print('NAcAL:', origen, '->', nacal or '(instalada)')
    print('casos:', len(todos))
    con_errores = {}
    with concurrent.futures.ThreadPoolExecutor(procesos) as grupo:
        tareas = [grupo.submit(ejecuta_caso, n, f, nacal, destino) for n, f in todos.items()]
        for tarea in concurrent.futures.as_completed(tareas):
            nombre, errores = tarea.result()
            if errores:
                con_errores[nombre] = errores
    (destino / 'ORIGEN').write_text(origen + '\n' + str(nacal or 'instalada') + '\n')
    print('casos con errores de ejecución:', len(con_errores))
    for nombre in sorted(con_errores):
        print('   ', nombre, '(sin resumen de PythonTeX)' if con_errores[nombre] < 0 else con_errores[nombre])
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
    c = ordenes.add_parser('comparar')
    c.add_argument('a')
    c.add_argument('b')
    ordenes.add_parser('casos')
    args = analizador.parse_args()
    if args.orden == 'ejecutar':
        ejecutar(args.etiqueta, args.nacal, args.procesos)
    elif args.orden == 'comparar':
        sys.exit(comparar(args.a, args.b))
    else:
        todos = casos()
        for nombre, fichero in todos.items():
            print(nombre, '<-', fichero)
        print(len(todos), 'casos')


if __name__ == '__main__':
    main()
