# Encargo desde las clases: enlace público y estable a la documentación de NAcAL

De la sesión de `MatematicasII-SRC`, 2026-11-22, con el visto bueno de
Marcos. Sale de la revisión de la Lección 3 de clase.

## Qué pasa

La última transparencia de la Lección 3 (`clases/lecciones/L03.tex`,
«Módulo nacal para Python») dice a los alumnos:

> Revise la implementación de las operaciones del álgebra matricial en
> el módulo nacal para Python que acompaña al curso: Sección 1.3 de la
> documentación (o estudie directamente el código).

y enlaza con

    https://github.com/mbujosab/nacallib/blob/master/doc/NACAL.pdf

Dos problemas:

1. Ese fichero ya no está en la rama `master` de `nacallib`. En `doc/`
   hay `ParaMisCompis.pdf` y `nacal-JDE-2024-*.pdf`; en la raíz,
   `nacal_src.pdf`. El enlace da error.
2. En `nacal_src.pdf` la Sección 1.3 es «Transformaciones elementales
   de un Sistema». Las operaciones algebraicas son la 1.2. El número de
   sección de la transparencia ya no vale.

Las lecciones 1 y 4 de clase enlazan solo con
`https://github.com/mbujosab/nacallib` («véase la documentación del
módulo»), sin fichero ni sección: esos enlaces funcionan.

## Qué se pide

1. Decir cuál es el enlace público que debe usar el material de clase
   para «la documentación» (¿`nacal_src.pdf` en `master`?, ¿otra
   dirección?) y si se puede contar con que no cambie durante el curso
   (enero a mayo de 2027).
2. Decir a qué parte de esa documentación debe remitir la transparencia
   para «la implementación de las operaciones del álgebra matricial»
   (suma, producto por escalar, producto punto, matriz por vector,
   producto de matrices, transposición). Si los números de sección
   pueden cambiar, un ancla con nombre o el título de la sección sirve
   mejor que el número.
3. Si conviene, una frase de una línea que el material pueda copiar
   para describir esa documentación a un alumno de primero.

## Mientras tanto

En la transparencia quitaré el número de sección y enlazaré con
`https://github.com/mbujosab/nacallib`, como las lecciones 1 y 4, hasta
tener la respuesta.

La respuesta, en
`MatematicasII-SRC/respuesta-enlace-documentacion-desde-nacal.md` o al
final de este fichero.

## Respuesta desde NAcAL (2026-11-22)

Tratado con Marcos, que pidió la modalidad más cómoda para el alumno.

**1. Enlace a la documentación.** Desde hoy la documentación se sirve
como página web:

    https://mbujosab.github.io/nacallib/nacal_src.pdf

El navegador abre el PDF directamente, con su índice y sus enlaces
internos, sin pasar por el visor de GitHub. Es el `nacal_src.pdf` de la
rama `master`; se actualiza solo cada vez que se sube un cambio. El
nombre y la dirección no cambiarán durante el curso. Hay además una
página de entrada, `https://mbujosab.github.io/nacallib/`, con enlaces
a la documentación, al libro, a PyPI y al código.

El enlace antiguo (`doc/NACAL.pdf`) no existe desde hace tiempo y no se
va a recuperar. `https://github.com/mbujosab/nacallib`, que usan las
lecciones 1 y 4, sigue valiendo.

**2. A qué parte remitir.** A la sección «Operaciones algebraicas
sobre Sistemas», dentro del capítulo «La clase Sistema». Tiene un ancla
con nombre, que no cambia aunque se renumeren las secciones:

    https://mbujosab.github.io/nacallib/nacal_src.pdf#target:OperacionesAlgebraicas

Hoy es la Sección 1.2, pero conviene citarla por el título y no por el
número. Contiene, en este orden: el operador selector por la derecha y
por la izquierda, la suma y la diferencia, el producto por un escalar,
y el producto de un `Sistema` por un `Vector` o una `Matrix` (ahí están
el producto punto, matriz por vector y el producto de matrices). La
transposición no está en esa sección: está en el capítulo «La subclase
BlockM».

**3. Una frase para el alumno.** Propuesta:

> La documentación de NAcAL es un PDF que explica, operación por
> operación, cómo está programado el módulo: cada apartado recuerda la
> definición del curso y muestra debajo el código de Python que la
> realiza.

Para la transparencia de la Lección 3 podría quedar así: «Revise cómo
están programadas las operaciones del álgebra matricial en el módulo
NAcAL: sección *Operaciones algebraicas sobre Sistemas* de su
documentación».

Un aviso: el PDF tiene 152 páginas y está escrito como programa
documentado, no como manual para alumnos. Sirve para quien quiera ver
el código; no hace falta para usar el módulo.
