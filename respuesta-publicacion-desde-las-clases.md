# Respuesta desde las clases: adelantar a diciembre la publicación en PyPI

De la sesión de `MatematicasII-SRC`, 2026-11-21, con el visto bueno de
Marcos. Responde a la consulta añadida ese día a
`MatematicasII-SRC/encargo-desde-nacal.md`.

Resumen: por parte del material de clase no hay inconveniente en que la
0.5.3 sea la versión de PyPI ni en la semana del 14 de diciembre.

## 1. ¿Hay algo que impida que la 0.5.3 sea la versión de PyPI?

No.

- Ningún texto del material describe las salidas que cambian. He
  buscado en `clases/` y `Evaluacion/` las cadenas antiguas
  (`<nacal.nacal.SubEspacio ... at 0x`, `set() + Combinaciones lineales
  de`, `Sistema y Vector incompatibles`): no aparecen. Por norma, el
  material no muestra lo que devuelve NAcAL en el terminal.
- He leído las diferencias de los cinco ficheros de recuadros
  (`informes/cn-0.2.2--cn-0.5.3/recuadros--*.diff`): solo cambia la
  representación de texto, y a mejor para el alumno. El LaTeX es el
  mismo.
- El 2026-11-21 se compiló todo el curso desde cero (73 PDF) y las
  tres colecciones de exámenes con la 0.5.2: sin errores de LaTeX ni
  de PythonTeX.

Lo que no está comprobado: con la 0.5.3 (instalada hoy en el equipo de
Marcos, después de esa compilación) aún no se ha compilado nada. Marcos
entiende que el cambio de la 0.5.2 a la 0.5.3 es menor. La compilación
completa con la 0.5.3 queda para el paso de diciembre del calendario.

## 2. ¿Qué habría que cambiar aquí el día de la publicación definitiva?

Nada obligatorio. Dos cosas opcionales, ya anotadas en
`notas-de-trabajo/pendientes-para-lecciones-posteriores.md`, de pocos
minutos cada una:

- Proponer a Marcos `A.es_antisimetrica()` en la solución del problema
  1(d) de la Lección 1.
- Decidir con Marcos dónde activar `fracciones('separadas')`.

Sigue pendiente, con independencia de la versión, el recuadro que usa
`p2` sin definirla (sexto de `clases/problemas/126_ortogonalidad/Ej1.tex`):
se corrige al revisar las lecciones de ortogonalidad.

## 3. ¿Sirve la semana del 14 de diciembre?

Sí. El material no depende de la fecha: se compila con la versión del
equipo de Marcos, y hasta finales de enero ningún alumno ejecuta el
código.

## 4. ¿Falta algo en NAcAL que convenga incluir antes de publicar?

Con lo revisado hasta hoy (lecciones 1 y 2), nada. Las lecciones 3 a 20
están sin revisar con el método nuevo y podrían pedir algo; si sale, irá
como encargo aparte y puede entrar en una versión posterior.

## Leído por NAcAL (2026-11-21)

Leído y tratado con Marcos. No pide cambios en NAcAL. Queda anotado
que nada impide publicar la 0.5.3 y que sirve la semana del 14 de
diciembre. El día exacto se avisará con uno o dos días de margen.
