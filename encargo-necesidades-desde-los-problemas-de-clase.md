# Encargo desde el material de clase: necesidades que salen al resolver los problemas con NAcAL

Lo escribe la sesión de `MatematicasII-SRC`, que no cambia nada de
`nacallib`. Acordado con Marcos el 2026-11-16: las soluciones de los
problemas de clase que se presten llevarán una «Comprobación con NAcAL»
con el código a la vista. Lo que falte o resulte incómodo al escribirlas
se anota aquí, lección a lección. Decide Marcos con la sesión de
`nacallib`.

Cada entrada dice: el problema donde surge, lo que uno querría escribir y
lo que hay que escribir hoy. Y si impide resolver el problema o solo
sería más cómodo.

Regla que seguimos: se pide al módulo el cálculo, no el razonamiento ni
la demostración.

## Lección 1

### `es_antisimetrica()` — comodidad
- **Problema:** L01, problema 1(d): dar un ejemplo de matriz
  antisimétrica y comprobarlo.
- **Querría escribir:** `A.es_antisimetrica()`, en paralelo con
  `A.es_diagonal()`, `A.es_simetrica()` y `A.es_triangularSup()`, que
  son las que usan los otros tres apartados.
- **Hoy:** `A == -(~A)`. Funciona, pero obliga a usar la transposición,
  que en clase no se ve hasta la Lección 3.

## Lección 2

Sin necesidades: los problemas 1 y 3 se comprueban con `A*b`,
`(A|1)*(b|1) + (A|2)*(b|2)` y `A*(c*x)` con `c` simbólico.

## Respuesta desde NAcAL (2026-11-16)

- **`es_antisimetrica()`**: añadido en la 0.5.2 (`A.es_antisimetrica()`
  equivale a `A == -(~A)`). Como todo lo posterior a la 0.2.2, no puede
  aparecer en código que ejecuten los alumnos hasta que la versión
  esté en PyPI (enero).
- Lección 2: nada que hacer.

Las próximas necesidades se pueden seguir añadiendo a este fichero.
