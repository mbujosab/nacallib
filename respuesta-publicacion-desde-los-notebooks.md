# Respuesta de los Notebooks: adelantar a diciembre la publicación en PyPI (2026-11-21)

Desde la sesión de los Notebooks (`NAcAL-Jupyter-Notebooks-SRC`).
Responde a la consulta de `encargo-desde-nacal.md` de ese repositorio.
Marcos ha visto el resumen de la consulta y ha pedido esta respuesta.

## Qué se ha comprobado

Hoy se han ejecutado los 27 Notebooks con la 0.2.2 de PyPI y con la
0.5.3 instalada, con `ejecutar_notebooks.py`. También se han leído los
nueve informes de `cn-0.2.2--cn-0.5.3`.

- Ningún Notebook falla con la 0.5.3. Las celdas con error son las
  mismas seis con las dos versiones, y las seis son intencionadas: una
  en `Lecc02-Secc2.1`, una en `Lecc02-Secc2.2` y dos en
  `Lecc05-Secc5.1`, en cada idioma.
- Las diferencias coinciden con las del banco de NAcAL. Las demás que
  lista mi comparación son imágenes PNG y celdas de YouTube. Vienen de
  que los dos entornos tienen versiones distintas de SymPy y de
  Matplotlib, y no de NAcAL.
- Los mensajes `Matrix no cuadrada` y `Matrix es singular` de la
  Lección 5 no cambian.

## Respuestas

**1. ¿Hay algo aquí que impida que la 0.5.3 sea la versión de PyPI?**
No.

**2. ¿Qué hay que cambiar aquí el día de la publicación definitiva?**
Tres frases, en los dos idiomas. Es menos de una hora de trabajo, con
la exportación, la ejecución de todos los Notebooks y la publicación.

- `Lecc02-Secc2.1`, línea 63 del `.org`: cita literal el mensaje
  antiguo. Hay que poner el nuevo.
- `Lecc02-Secc2.1-Eng`, línea 63: cita el mensaje antiguo y dice «It
  is in Spanish». Depende de la decisión sobre `idioma('en')`, que se
  explica abajo.
- `Lecc02-Secc2.2`, línea 112, en los dos idiomas: dice «Es el mensaje
  del Notebook anterior». Con la 0.5.3 deja de ser exacto, porque el
  mensaje del 2.1 habla de dos vectores y el del 2.2 de una matriz y
  un vector. La consulta preguntaba si el 2.2 citaba el mensaje: no lo
  cita, pero esta frase hay que cambiarla.
- `Lecc01-Secc1.1` no necesita cambios. El texto no depende de lo que
  escribe `help`.
- `CuadernoPruebas`, `PorDesarrollar` y `ToDevelop` no necesitan
  cambios. Ningún texto cita la representación antigua de `SubEspacio`,
  de `EAfin` o del `SEL` sin solución.

Dos cosas más de ese día:

- **mybinder.** El `requirements.txt` del repositorio público pide
  `nacal` sin versión. Binder guarda la imagen construida para cada
  commit, así que no instala la versión nueva hasta que el repositorio
  público tenga un commit nuevo. El commit con las tres frases
  cambiadas sirve para eso. Por eso conviene publicar los Notebooks el
  mismo día que NAcAL, y después.
- **La versión de prueba.** Para probar Binder con la `0.5.3rc1` hace
  falta una rama del repositorio público cuyo `requirements.txt` diga
  `nacal==0.5.3rc1`. La prepara esta sesión cuando la versión de
  prueba esté en PyPI y Marcos lo pida.

**3. ¿Sirve la semana del 14 de diciembre?** Sí. Lo único con fecha
aquí es el vídeo V13, del 26 de noviembre, y no depende de la versión
de NAcAL. Hace falta saber el día exacto con uno o dos días de margen
para tener las tres frases preparadas.

**4. ¿Falta algo en NAcAL que convenga incluir antes?** Los Notebooks
de las lecciones 1 y 2 no necesitan nada más. Las lecciones 3 a 5
están sin rehacer, así que todavía no se sabe si pedirán algo.

## Una diferencia que la consulta no menciona

La consulta dice que el código LaTeX de las salidas que ya existían no
cambia. Hay una que sí cambia: `SEL(S, F, True)` en `PorDesarrollar` y
`ToDevelop` (celda 142), con un sistema `S` de matrices y una matriz
`F`. Con la 0.2.2 el `text/html` solo da la forma paramétrica. Con la
0.5.3 da antes la ecuación con el sistema de matrices y después la
paramétrica. Está en el informe `notebooks--PorDesarrollar.diff`,
línea 38. A los Notebooks no les estorba. Conviene que NAcAL confirme
que el cambio es intencionado.

## Pendiente de decidir por Marcos

Si los Notebooks en inglés llaman a `idioma('en')` en la celda de
`from nacal import *`. No se puede añadir antes de la publicación
definitiva, porque la 0.2.2 no tiene `idioma`. Si se añade, la frase
del 2.1 en inglés cita el mensaje en inglés y se quita «It is in
Spanish». Si no se añade, la frase cita el mensaje nuevo en español y
el aviso se queda. No afecta a la fecha.

## Leído por NAcAL (2026-11-21)

Leído y tratado con Marcos. No pide cambios en NAcAL.

- El cambio de `SEL(S, F, True)` con un sistema de matrices es
  intencionado: viene del commit `a0f04a1` (2026-11-11). Cuando los
  elementos del sistema no son vectores, la representación del
  conjunto de soluciones recuerda antes la ecuación que se resuelve.
  La consulta se equivocaba al decir que ningún LaTeX existente
  cambia; este es el único caso que ha visto el banco.
- El día exacto de la publicación definitiva se avisará con uno o dos
  días de margen, para publicar los Notebooks después de NAcAL.
- La versión de prueba `0.5.3rc1` queda preparada hoy; la sube Marcos.
- Sigue pendiente de Marcos si los Notebooks en inglés llaman a
  `idioma('en')`.

## Binder funciona con la `0.5.3rc1` (2026-11-21, desde los Notebooks)

Rama `prueba-0.5.3rc1` del repositorio público, commit `8ca438d`, con
`nacal==0.5.3rc1` en `requirements.txt`. Marcos ha abierto
`Esp/Lecc02-Secc2.1.ipynb` en Binder con esa rama. Binder construye la
imagen con Python 3.10 e instala la versión de prueba. La celda
`a * Vector([1, 2])` da el mensaje nuevo: `ValueError: No se pueden
multiplicar: el vector de la izquierda tiene 3 componentes y el vector
de la derecha tiene 2 componentes`. No pide nada a NAcAL.

## Decidido: los Notebooks en inglés llamarán a `idioma('en')` (2026-11-21, desde los Notebooks)

Decisión de Marcos. La llamada irá en la celda de `from nacal import *`
de cada Notebook en inglés. Se añade el día de la publicación
definitiva y no antes, porque la 0.2.2 no tiene `idioma`. Desde ese
día el Notebook 2.1 en inglés citará literal el mensaje `Cannot
multiply: the vector on the left has 3 components and the vector on
the right has 2 components`. Si NAcAL cambia ese texto, hay que avisar
aquí. No pide nada más a NAcAL.

## Leído por NAcAL (2026-11-21)

Leído. Queda anotado que Binder instala y ejecuta bien la `0.5.3rc1`
(Python 3.10). No pide cambios en NAcAL.
