# Encargo desde las clases: el código que los alumnos ven y pueden ejecutar por su cuenta

Escrito el 2026-11-16 desde la sesión de `MatematicasII-SRC`. Responde a
la petición del final de `MatematicasII-SRC/encargo-desde-nacal.md`
(lista de lo que los alumnos ejecutan por su cuenta, para añadirlo al
banco de pruebas antes de publicar en PyPI). Aquí solo se anota.

## Qué hay en el material de clase

El material de clase no tiene cuadernos. Lo que un alumno puede copiar y
ejecutar son los recuadros de código de los apuntes: entornos
`pyverbatim`, que **se muestran pero PythonTeX no ejecuta**. Por eso no
están en los `.pytxcode` y el banco no los cubre.

Ficheros con recuadros (rutas desde `MatematicasII-SRC/`; entre
paréntesis, el número de recuadros):

    clases/lecciones/L05.tex (1)  L08.tex (1)  L15.tex (1)  L16.tex (1)  L19.tex (1)
    clases/problemas/010_OperacionesVectoresMatrices/Ej.tex (4)
    clases/problemas/010_Geometria_ec_lineales/Ej.tex (2)
    clases/problemas/020_Elimin_Gaussiana/Ej.tex (11)
    clases/problemas/030_Multiplicacion_matricial/Ej.tex (4)
    clases/problemas/040_Matriz_inversa/Ej.tex (9)
    clases/problemas/070_Eliminacion_Gaussiana_de_Sis_Homogeneos/Ej.tex (8)
    clases/problemas/080_Eliminacion_Gauss-Jordan_de_Sis_Homogeneos/Ej.tex (3)
    clases/problemas/095_Rouche-Frobenius/Ej.tex (3)
    clases/problemas/126_ortogonalidad/Ej.tex (5)  Ej1.tex (9)  Ej2.tex (5)
    clases/problemas/130_propiedades_determinantes/Ej-2dias.tex (1)
    clases/problemas/160_Autovalores_y_Autovectores/Ej.tex (2)
    clases/problemas/170_Diagonalizacion/Ej.tex (2)
    clases/problemas/180_Matrices_simetricas/Ej.tex (2)
    clases/problemas/190_Matrices_definidas--Mat_Simetricas/Ej.tex (2)

Para sacarlos: el texto entre `\begin{pyverbatim}` y `\end{pyverbatim}`.
Cada recuadro presupone `from nacal import *`; algunos usan `sympy` y
`display`.

## Límites

- La lista es del 2026-11-16, por búsqueda del entorno; puede cambiar
  durante la revisión del material (lección a lección, en curso).
- Solo he ejecutado los recuadros de las dos primeras carpetas de
  problemas (lecciones 1 y 2): funcionan con la 0.2.2 y con la 0.5.0.
  El resto está sin ejecutar; los iré ejecutando al revisar cada
  lección.
- Los cuadernos que usan los alumnos están en el repositorio de los
  Notebooks: esa parte de la lista le toca a su sesión.

## Respuesta desde NAcAL (2026-11-16)

Tratado. `tests/banco_material.py` ejecuta ahora también los recuadros
`pyverbatim` de `MatematicasII-SRC/clases` (21 ficheros, 77 recuadros),
fichero a fichero y como lo haría un alumno en un terminal, y con
`--notebooks` los Notebooks con el guion de su repositorio.

Resultado con la 0.2.2 (PyPI) frente a la 0.5.2: los recuadros dan lo
mismo salvo la representación de texto de `SubEspacio`, `EAfin` y de
un `SEL` sin solución (cambios ya conocidos). Un recuadro falla con
todas las versiones: el sexto de
`clases/problemas/126_ortogonalidad/Ej1.tex` (`NameError: name 'p2' is
not defined`); anotado en `MatematicasII-SRC/encargo-desde-nacal.md`.
