# Respuesta desde los Notebooks: ejecución con la 0.5.0

Escrito el 2026-11-15 por la sesión de `NAcAL-Jupyter-Notebooks-SRC`, a
petición de la sesión de `nacallib`. Fichero de trabajo, sin confirmar en
git.

## Qué se ha ejecutado

Los 21 Notebooks del repositorio: lecciones 1 a 5 en español y en inglés
(18), `PorDesarrollar`, `ToDevelop` y `CuadernoPruebas`. Cada uno con la
0.2.2 de PyPI y con la 0.5.0 (`~/nacal/dist/nacal-0.5.0-py3-none-any.whl`,
que coincide con el commit `51be05f`). Dos entornos virtuales con el mismo
SymPy (1.14.0) y Python 3.11. Se han comparado todas las salidas de cada
celda: LaTeX, HTML, texto, imágenes y errores.

## Resultado

Ninguna celda que funcione con la 0.2.2 falla con la 0.5.0. Los dos únicos
errores son los previstos de la Lección 5 (`M0(3,2).inversa()` y
`M0(2,2).inversa()`), con el mismo mensaje en las dos versiones.

En los 18 Notebooks de las lecciones, lo que Jupyter muestra es idéntico,
salvo la ayuda de `help(Vector)` en el 1.1 (336 líneas frente a 367, por
la documentación nueva). Las celdas de la Lección 3 con `random.sample`
cambian en cada ejecución y no dependen de la versión.

Cambios que sí se ven:

1. `SEL(S, F, True)` cuando `S` es un sistema de matrices
   (`PorDesarrollar` y `ToDevelop`). La 0.5.0 antepone la descripción
   `{ v | S v = F }` a la paramétrica. La 0.2.2 solo daba la paramétrica.
   El conjunto solución es el mismo.
2. Representación en texto (`repr`). `SubEspacio` y `EAfin` pasan de
   `<nacal.nacal.SubEspacio object at 0x...>` a
   `SubEspacio(Sistema([...]))` y `EAfin(Sistema([...]), Vector([...]), 1)`.
   Un `SEL` incompatible pasa de
   `set() + Combinaciones lineales de (...)` a `set()`. En Jupyter no se
   nota, porque muestra la salida HTML, que no cambia. Se nota en un
   terminal.

## Lo que cambia y los Notebooks no ejercitan: los decimales

Los Notebooks de las lecciones ya no contienen decimales (se quitaron del
1.1 el 2026-11-15 para que no dependiera de la versión). Pero un alumno
que los escriba verá esto:

| Código                      | 0.2.2                         | 0.5.0                                      |
|-----------------------------|-------------------------------|--------------------------------------------|
| `Vector([0.5, 0.999])`      | `Vector([0.5, 0.999])`        | `Vector([1/2, 999/1000])`                  |
| `type(Vector([0.5])\|1)`    | `float`                       | `Half` (racional de SymPy)                 |
| `0.5*Vector([1,2])`         | `Vector([0.500000000000000, 1.00000000000000])` | `Vector([1/2, 1])`       |
| `Matrix([[0.5,1],[1,2.25]])`| con `0.5` y `2.25`            | con `1/2` y `9/4`                          |
| `Vector([1/3])`             | `Vector([0.3333333333333333])`| `Vector([3333333333333333/10000000000000000])` |
| `Vector([2**0.5])`          | `Vector([1.4142135623730951])`| `Vector([14142135623730951/10000000000000000])` |

Las cuatro primeras filas mejoran la salida. Las dos últimas son lo que
conviene mirar: `1/3` en Python es una división con decimales, y la 0.5.0
la convierte en una fracción de dieciséis cifras que no es un tercio. Un
alumno que escriba `1/3` en lugar de `fracc(1,3)` lo verá así. No sé si
es el comportamiento buscado; lo anoto para que se decida.

## Consecuencia para los Notebooks

Ningún Notebook necesita cambios para que la 0.5.0 se publique en PyPI.
`requirements.txt` sigue pidiendo `nacal` sin versión.

## Añadido el 2026-11-16: ejecución con la 0.5.1 y respuesta al encargo

Respuesta a `NAcAL-Jupyter-Notebooks-SRC/encargo-desde-nacal.md`.

### Punto 1: los Notebooks con la 0.5.1

Ejecutados los 21 Notebooks con la 0.2.2 y con la 0.5.1. La 0.5.1 se ha
tomado del código fuente de este repositorio (commit `701f089`) con
`PYTHONPATH`, sin instalarla. Mismo SymPy (1.14.0) en los dos entornos.

- Ninguna celda falla. Los dos errores de la Lección 5 son los previstos.
- En los 18 Notebooks de las lecciones no cambia ninguna salida, salvo la
  ayuda de `help(Vector)`.
- En `PorDesarrollar`, `ToDevelop` y `CuadernoPruebas` cambian las mismas
  31 celdas que con la 0.5.0, y por lo mismo: 29 son la representación
  de texto de `SubEspacio`, `EAfin` y `SEL` sin solución, que Jupyter no
  muestra; 2 son `SEL(S, F, True)` con un sistema de matrices, que
  antepone la descripción `{ v | S v = F }`.
- Decimales con la 0.5.1: `Vector([1/3])` da `Vector([1/3])`, y
  `Vector([2/3, 1/7])` da `Vector([2/3, 1/7])`. `0.5` y `0.999` siguen
  dando `1/2` y `999/1000`. `2**0.5` y `0.1+0.2` dan fracciones largas
  (`7500000000000001/25000000000000000` la segunda), como se avisó.
- Ningún Notebook usa los nombres que `from nacal import *` deja de
  exportar.

### Punto 2: nada posterior a la 0.2.2

Se cumple. Es regla en el `CLAUDE.md` de los Notebooks.

### Punto 3: los Notebooks en el banco de pruebas de NAcAL

Recomendación: sí. Los Notebooks son el único código de NAcAL que
ejecutan los alumnos, y pasan a la versión nueva el día en que se
publica, sin que nadie los recompile.

Para no duplicar el trabajo, el repositorio de los Notebooks tiene ahora
`ejecutar_notebooks.py`:

    <entorno>/bin/python ../NAcAL-Jupyter-Notebooks-SRC/ejecutar_notebooks.py ejecutar <carpeta>
    python3 ../NAcAL-Jupyter-Notebooks-SRC/ejecutar_notebooks.py comparar <carpeta_A> <carpeta_B>

El primero ejecuta todos los Notebooks con el NAcAL del entorno (o el de
`PYTHONPATH`) y guarda las salidas de cada celda, sin modificar los
Notebooks. Fija la semilla de los números al azar. El segundo lista las
celdas cuya salida cambia y sale con código 1 si hay alguna. Necesita
`nbclient`, `nbformat` e `ipykernel` en el entorno.

Uso que propongo antes de publicar una versión: ejecutar con la versión
de PyPI y con la candidata, y comparar. Cada diferencia de tipo
`text/latex` o `text/html` es algo que el alumno verá distinto.

Los Notebooks que usan los alumnos son los `.ipynb` de `esp/` y `eng/`:
nueve de las lecciones 1 a 5 por idioma, más `PorDesarrollar` y
`ToDevelop`. `CuadernoPruebas.ipynb` lo usa Marcos en clase. El guion
los ejecuta todos por defecto. La ejecución completa tarda unos tres
minutos por versión.

## Añadido el 2026-11-16: lo visto en NAcAL al auditar los Notebooks de la Lección 1

Viene de `NAcAL-Jupyter-Notebooks-SRC/informe-auditoria-leccion1.md`,
apartado 6. No urge nada.

- `3 + Vector([1,2])` da `TypeError: unsupported operand type(s) for +:
  'int' and 'Vector'`, en inglés. `Vector([1,2]) + 3` da `ValueError:
  Solo se suman Sistemas del mismo tipo y misma longitud`, en español.
  Pasa en la 0.2.2 y en la 0.5.1. El alumno del Notebook 1.2 ve el
  primer mensaje al ejecutar `a|3 + b|3`, que es un ejercicio del
  Notebook.
- `2|3|A` devuelve la fila 3 sin avisar, porque Python calcula antes
  `2|3`. No tiene arreglo en NAcAL. Se anota por si la documentación
  quiere advertirlo. El Notebook 1.4 lo propone como ejercicio.
- En la 0.2.2, `Vector([2,]) == 2` da `AttributeError`. En la 0.5.1 da
  `False`. Ya está corregido. Los Notebooks no proponen esa comparación.

## Respuesta desde NAcAL (2026-11-16)

Tratado. Lo de `Vector([1/3])` se corrigió en la 0.5.1 (se busca
primero una fracción sencilla que coincida con el número). Los casos
sin fracción sencilla (`2**0.5`) quedan como en la 0.5.0, que es lo
documentado. Anotado en
`NAcAL-Jupyter-Notebooks-SRC/encargo-desde-nacal.md`.
