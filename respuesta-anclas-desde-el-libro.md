# Respuesta desde el libro: anclas con nombre

Escrita el 2026-11-22 desde la sesión del libro, que no ha cambiado nada
de este repositorio. Sin confirmar en git.

Hecho. Cuatro anclas con nombre en `partes/Cap01-OperacionesVectoresYMatrices.tex`:

| Ancla numérica | Ancla con nombre |
|---|---|
| `chapter.1` | `target:LeccVectoresYMatrices` |
| `section.1.1` | `target:VectoresRn` |
| `section.1.2` | `target:SumaVectoresYProductoPorEscalares` |
| `section.1.5` | `target:SumaMatricesYProductoPorEscalares` |

Cada una cae a la misma altura de la página que su ancla numérica
(comprobado con `pdfinfo -dests`), así que el enlace muestra el título.
No cambian el texto ni la paginación. No se renombrarán. Estarán en la
web en la próxima publicación de `libro.pdf`; hasta entonces siguen
valiendo las numéricas.

Las anclas de las tres diagonalizaciones se pondrán al revisar esas
lecciones y se anotarán aquí.

## Publicado (2026-11-22)

El `libro.pdf` con las cuatro anclas con nombre está publicado en la
web (commit `1bcb565` del repositorio público). Ya se pueden usar.

## Leído por NAcAL (2026-11-22)

Hecho. `nacal_src.org` usa ya las cuatro anclas con nombre en sus siete
enlaces a la Lección 1; no queda ningún enlace con ancla numérica.
`tests/test_enlaces_al_libro.py` las comprueba. Quedan pendientes las
anclas de las tres diagonalizaciones, cuando el libro revise esas
lecciones.
