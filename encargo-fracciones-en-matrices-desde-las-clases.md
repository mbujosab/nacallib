# Encargo desde el material de clase: fracciones dentro de matrices

Escrito el 2026-11-15 desde la sesión del repositorio
`MatematicasII-SRC`, tras hablarlo con Marcos. Siguiendo
`coordinacion-entre-repositorios.md`, aquí solo se anota: esta sesión
no ha cambiado nada de `nacallib`.

**Qué se pide:** que la sesión de `nacallib` comente estas propuestas
con Marcos y decidan entre los dos. No es un encargo de implementar
nada antes de hablarlo.

## El problema

Cuando una matriz tiene fracciones en filas contiguas, `latex()` las
escribe con `\frac` dentro de un `array` y las filas quedan demasiado
juntas: el denominador de una fila casi toca el numerador de la fila
de abajo. Marcos lo ha visto impreso y no le parece aceptable.

Dónde apareció: `Evaluacion/preguntas/Intermedios/2026-intermedio2-C.tex`
(cuestionario `2026-Quiz2-C`, ejercicio 5), con

    A = Matrix([Vector([0.5, 0.5, 0]), Vector([0, 0.9, 0.1]), Vector([0, 0, 1])])

Desde la 0.3.0 los decimales se convierten en racionales, y
`latex(A)` da

    \left[ \begin{array}{ccc}\frac{1}{2}&0&0\\\frac{1}{2}&\frac{9}{10}&0\\0&\frac{1}{10}&1\\ \end{array} \right]

Comprobado con la 0.4.1 instalada. Para verlo impreso basta compilar
esa salida en un documento cualquiera.

En esa pregunta ya está resuelto sin tocar NAcAL: la matriz del
enunciado se ha escrito a mano con decimales, que es como se puso en
el examen. Lo que queda abierto es el caso general, porque volverá a
salir en cualquier matriz con fracciones, en las transparencias, en
los exámenes y en el libro.

## Las tres propuestas de Marcos

1. **Más espacio entre filas** al pintar la matriz. Marcos duda de si
   quedará raro.
2. **Fracciones con barra inclinada**, como en `1/2`.
3. **Que NAcAL permita activar números de coma flotante.**

## Condiciones que ya están acordadas

- Los cambios de NAcAL que afecten a las salidas deben ser opcionales;
  la salida por defecto no cambia sin decidirlo con Marcos
  (`coordinacion-entre-repositorios.md`).
- La salida por defecto la usan también el libro y el resto del
  material de clase.
- La conversión de decimales en racionales exactos fue decisión de
  Marcos (ver `MatematicasII-SRC/encargo-desde-nacal.md`).

## Valoración de la sesión del material de clase

Marcos dejó a mi criterio incluirla. Va al final y separada para que
la sesión de `nacallib` se forme antes su propia opinión, que conoce
el código y yo no. Es una opinión, no un acuerdo con Marcos.

- **Espacio entre filas:** me parece la solución tipográfica habitual.
  No creo que quede raro si se aplica solo a las matrices que contienen
  alguna fracción; aplicado a todas, las matrices de enteros quedarían
  demasiado abiertas. No sé cómo de fácil es detectarlo en el código ni
  si conviene hacerlo por matriz o por fila.
- **Barra inclinada:** se lee bien con números pequeños y no cambia la
  altura de la fila. Me parece peor con signos y con expresiones
  simbólicas, como `-(a+1)/2`. La veo como alternativa opcional.
- **Coma flotante:** no me gusta si significa calcular con decimales,
  porque se pierde la aritmética exacta. Sería distinto una opción solo
  de presentación: guardar racionales y mostrarlos como decimal cuando
  el decimal es finito. Habría que decidir qué se hace con `1/3`.

No he mirado cómo se ven las matrices con fracciones en el libro ni si
allí ya se resuelve de algún modo.

## Respuesta desde NAcAL (2026-11-16)

Tratado. Acordado con Marcos y hecho en la 0.5.0: función `fracciones`
con los modos `'normal'` (por defecto, nada cambia), `'separadas'`
(más espacio entre las filas de matrices y vectores con fracciones;
0.6ex o la longitud que se indique) e `'inclinadas'` (barra inclinada).
La tercera propuesta (mostrar decimales) no se ha hecho. Detalles en
`MatematicasII-SRC/encargo-desde-nacal.md` y en `nacal_src.org`,
apartado «Presentación de las fracciones». Queda una decisión de
Marcos: si algún modo pasa a ser el de por defecto.
