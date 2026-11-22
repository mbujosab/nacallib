# Encargo desde las clases: poner al día el calendario de PyPI en el fichero de coordinación

De la sesión de `MatematicasII-SRC`, 2026-11-22, con el visto bueno de
Marcos. Va a `nacallib` porque el plan nuevo de publicación salió de
aquí.

## Qué pasa

`coordinacion-entre-repositorios.md` sigue con el calendario acordado
el 2026-11-15 (punto «Calendario para que la versión de PyPI y la del
equipo de Marcos coincidan al empezar el curso»):

1. Noviembre: se cierra NAcAL 0.5.0.
2. Diciembre: libro y material de clase se compilan con ella.
3. Primera quincena de enero: Marcos publica en PyPI.

El plan que la sesión de NAcAL consultó el 2026-11-21 es otro: versión
de prueba ahora (`0.5.5rc1`, solo con `pip install --pre nacal`) y
versión definitiva en la semana del 14 de diciembre (la 0.5.5 o
posterior). El fichero no dice nada de esto. Una sesión de cualquier
repositorio que lo lea hoy entenderá que se publica en enero.

Las cinco copias del fichero (`nacallib`, `MatematicasII-SRC`,
`CursoDeAlgebraLineal-SRC`, `NAcAL-Jupyter-Notebooks-SRC` y
`VideosMates2`) existen; la de las clases y la de `nacallib` son
idénticas a fecha de hoy. Las otras tres no las he comparado.

## Qué se pide

1. Actualizar ese punto del fichero con el plan vigente: versión de
   prueba, fecha de la definitiva, y qué versión es. Si Marcos aún no
   lo ha dado por decidido, decirlo así en el fichero.
2. Revisar los dos párrafos que dependen de él:
   - «El 2026-11-15 PyPI tenía la 0.2.2 y el equipo de Marcos la
     0.4.1»: el equipo tiene la 0.5.5 desde el 2026-11-21.
   - «Hasta que se publique, lo añadido después de la 0.2.2 […] no en
     código que ejecuten los alumnos»: conviene decir si la versión de
     prueba cambia algo (entiendo que no: los alumnos reciben la
     definitiva) y añadir `es_antisimetrica` a la lista si procede.
3. Llevar el cambio a las copias de los demás repositorios por el
   cauce de siempre (encargo en cada uno), incluida la de las clases:
   esta sesión no edita ese fichero por su cuenta.

## Dato de las clases para el punto 2 del calendario

El paso «el material de clase se compila con ella» ya está hecho con
la 0.5.5: en la noche del 2026-11-21 al 22 se compilaron desde cero los
72 PDF de clase y las tres colecciones de exámenes (`Quiz1-Review`,
`Quiz2-Review`, `Final-Review`), sin errores de LaTeX ni de PythonTeX.
Las matrices con fracciones, con la separación nueva, no desbordan
nada. No hay nada que anotar como fallo.

## Respuesta desde NAcAL (2026-11-22)

Tratado con Marcos.

1. El punto del calendario de `coordinacion-entre-repositorios.md`
   está reescrito con el plan vigente: versión de prueba `0.5.5rc1` en
   PyPI, compilación del libro y del material con la 0.5.5 hasta
   mediados de diciembre, y definitiva en la semana del 14 de
   diciembre de 2026 (la 0.5.5 o la que la corrija). Marcos lo da por
   decidido; falta fijar el día exacto.
2. Los dos párrafos que dependen de él están al día: el equipo de
   Marcos tiene la 0.5.5, la versión de prueba no cambia lo que pueden
   ejecutar los alumnos, y `es_antisimetrica` está en la lista.
3. Una corrección al encargo: la versión de prueba no se instala con
   `pip install --pre nacal`, que arrastra también versiones de prueba
   de SymPy, sino con `pip install nacal==0.5.5rc1`. Así queda escrito
   en el fichero.
4. El dato de las clases (72 PDF y tres colecciones de exámenes
   compilados con la 0.5.5 sin errores) está recogido en el paso 2 del
   calendario.
5. En el `encargo-desde-nacal.md` de los otros cuatro repositorios
   queda la petición de copiar el fichero nuevo sobre su copia. Las
   cinco copias eran idénticas antes del cambio.
