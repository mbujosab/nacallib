"""Los enlaces de la documentación (nacal_src.org) al libro deben existir.

Cada enlace `{{{libro(ancla,texto)}}}` apunta a un ancla del PDF del libro.
Se comprueba contra el repositorio del libro, si está junto a este:

- las anclas con nombre deben estar definidas con `\\hypertarget{ancla}`
  en los fuentes del libro (no cambian aunque se renumeren las secciones);
- las anclas numéricas (`section.4.2`, etc.) deben figurar en `libro.out`.
  Son provisionales: existen mientras no cambie la numeración, pero nada
  garantiza que sigan apuntando a la misma sección.
"""
import pathlib
import re

import pytest

RAIZ  = pathlib.Path(__file__).resolve().parent.parent
ORG   = RAIZ / 'nacal_src.org'
LIBRO = RAIZ.parent / 'CursoDeAlgebraLineal-SRC'

NUMERICA = re.compile(r'^(chapter|section|subsection|subsubsection|appendix)\.')


def anclas_usadas():
    texto = ORG.read_text(encoding='utf-8')
    usadas = set(re.findall(r'\{\{\{libro\(([^,)]+),', texto))
    return sorted(usadas - {'ancla del pdf'})


def test_hay_enlaces_al_libro():
    assert anclas_usadas()


@pytest.mark.skipif(not (LIBRO / 'partes').is_dir(), reason='no está el repositorio del libro')
def test_las_anclas_existen_en_el_libro():
    fuentes = ''.join(f.read_text(encoding='utf-8', errors='replace') for f in LIBRO.glob('partes/**/*.tex'))
    con_nombre = set(re.findall(r'\\hypertarget\{([^}]+)\}', fuentes))
    salida = LIBRO / 'libro.out'
    numericas = set(re.findall(r'\\BOOKMARK \[\d+\]\[[^\]]*\]\{([^}]+)\}',
                               salida.read_text(encoding='latin-1'))) if salida.exists() else None

    faltan = []
    for ancla in anclas_usadas():
        if NUMERICA.match(ancla):
            if numericas is not None and ancla not in numericas:
                faltan.append(ancla)
        elif ancla not in con_nombre:
            faltan.append(ancla)
    assert not faltan, 'anclas que no existen en el libro: ' + ', '.join(faltan)
