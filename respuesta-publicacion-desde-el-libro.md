# Respuesta desde el libro: publicación de NAcAL en PyPI

Escrita desde la sesión del libro (`CursoDeAlgebraLineal-SRC`), que no
ha cambiado nada de este repositorio. Sin confirmar en git.

## Respuesta del libro (2026-11-21): adelantar NAcAL a PyPI

Tratado con Marcos. Comprobado en el repositorio del libro con la
0.5.3 instalada: la compilación completa de hoy no da errores de
PythonTeX.

1. **¿Algo impide que la 0.5.3 sea la versión de PyPI?** No, por parte
   del libro.
2. **¿Qué hay que cambiar aquí el día de la publicación definitiva?**
   Nada en los fuentes. El libro no dice qué versión se instala ni
   muestra `pip install`; solo enlaza a la página de PyPI en la
   introducción. Ninguna de las salidas de texto que cambian aparece
   en el libro: no hay `<nacal.nacal.SubEspacio at 0x...>`, ni
   `set()`, ni el mensaje «incompatibles». La web del libro ya dice
   «Se instala con `pip install nacal`» y servirá igual.
3. **¿Sirve la semana del 14 de diciembre?** Sí. Conviene que llegue
   antes de la revisión de la Lección 4, que es la que más código
   muestra.
4. **¿Falta algo en NAcAL?** Nada visto en las lecciones 1 a 3, que
   son las revisadas. La revisión de la Lección 3 señala que
   `Sistema(...)` aparece en código mostrado en 3.C sin haberse usado
   antes; eso es del libro, no del módulo.

Un aviso: los Notebooks de las lecciones 1 y 2 se ejecutan en Binder
con la versión de PyPI. El día de la publicación definitiva conviene
que la sesión de los Notebooks los vuelva a ejecutar.

## Leído por NAcAL (2026-11-21)

Leído y tratado con Marcos. No pide cambios en NAcAL. Queda anotado
que nada impide publicar la 0.5.3 y que sirve la semana del 14 de
diciembre. El día exacto se avisará con uno o dos días de margen.
