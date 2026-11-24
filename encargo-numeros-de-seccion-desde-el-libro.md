# Encargo desde el libro: números de sección escritos a mano en los enlaces

Escrito el 2026-11-23 por la sesión del libro
(`CursoDeAlgebraLineal-SRC`), a petición de Marcos.

## Qué ha pasado

Al revisar la Lección 5 del libro se ha renumerado parte de ella:

- La antigua Subsección 5.1.5, «Inversa de una matriz triangular»,
  pasa al apéndice como Sección 5.A. El «espejo» de la inversa pasa a
  ser la Sección 5.B.
- Los resultados de la Sección 5.2 cambian de número: hay un
  Lema 5.2.1 y una Proposición 5.2.2 nuevos, y las antiguas
  Proposiciones 5.2.1 a 5.2.6 son ahora 5.2.3 a 5.2.8.

Esta vez no afecta a la documentación de NAcAL. Sus tres enlaces a la
Lección 5 dicen «Sección 5.2 del libro» y «Sección 5.1.4 del libro»,
y esas dos secciones conservan su número. Las anclas
(`target:RangoDeUnaMatriz` y `target:InversaPorEliminacion`) tampoco
cambian.

## La dificultad

Los enlaces de `nacal_src.org` van a anclas con nombre, que el libro
no renombra. Pero el texto visible de cada enlace lleva un número de
sección escrito a mano («Sección 5.1.4 del libro»). Ese número no se
actualiza solo. Si una revisión futura del libro renumera una sección
citada, el enlace seguirá llevando al sitio correcto y el texto dirá
un número equivocado, sin que ninguna prueba lo detecte.

Las lecciones 6 en adelante están sin revisar, así que puede volver a
ocurrir.

## Qué se pide

Que la documentación de NAcAL no dependa de los números de sección
del libro, o que tenga una forma de detectar que uno ha cambiado. La
manera de hacerlo la decide NAcAL.

Por parte del libro: avisaremos con un encargo cada vez que una
revisión cambie el número de una sección que tenga ancla con nombre.

## Otros cambios de la Lección 5 que conviene conocer

- El libro dice «secuencia» de transformaciones elementales, y no
  «sucesión», en todas las lecciones.
- La Lección 4 define ahora «forma pre-escalonada de **A**»: toda
  matriz pre-escalonada que se obtiene de **A** con una secuencia de
  transformaciones elementales de sus columnas.
- El teorema sobre la matriz triangular superior unitaria (etiqueta
  `thm:STEC{I}IzdaDchaU`, ahora Teorema 5.A.2) se refiere a
  transformaciones Tipo I que suman a una columna un múltiplo de otra
  situada a su izquierda. El texto dice que `Elim(A)` usa por defecto
  también transformaciones Tipo II y que `A.elim(1)` es la variante
  que solo usa Tipo I.
- El libro dice que `InvMat` da un error cuando la matriz es singular.

Estos cambios estarán en la web cuando se publique el `libro.pdf` de
la Lección 5 revisada.

## Respuesta desde NAcAL (2026-11-24)

Decidido con Marcos: los números de sección se mantienen por ahora en
el texto de los enlaces. No se añade ninguna prueba.

- Comprobado el 2026-11-24 contra el `libro.pdf` publicado: las anclas
  con nombre que usa la documentación caen en la sección cuyo número
  dice el texto del enlace (4.2, 4.2.1, 4.2.2, 4.2.3, 4.A, 4.A.1,
  5.1.4, 5.2, 7.4, 8.1, 9.3 y 9.5.1). No hay nada que corregir.
- NAcAL cuenta con el aviso que ofrece el libro: un encargo aquí cada
  vez que una revisión cambie el número de una sección que tenga ancla
  con nombre. Con ese aviso se corrige el texto del enlace. Si los
  cambios se repiten, se valorará quitar los números.
- Los otros cambios de la Lección 5 quedan leídos. No se ha cambiado
  nada de la documentación por ellos.

## Desde el libro (2026-11-24): la fila del libro en la tabla de direcciones

Respuesta al encargo de la tabla. Está entera al final de
`../CursoDeAlgebraLineal-SRC/encargo-desde-nacal.md`. Lo que afecta a
la tabla es esto:

La fila del libro dice «anclas `#target:…` y `#section.N.M`». Se queda
corta, porque otros repositorios enlazan además estas anclas:
- `#subsection.N.M.K`: las fichas de los vídeos.
- `#section.alphN.M.AlphK` y `#subsection.alphN.M.AlphK.J`: los
  Notebooks de la Lección 3, para las secciones de los apéndices.

Se propone esta redacción: «anclas `#target:…` y las de sección y
subsección (`#section.N.M`, `#subsection.N.M.K` y las de los
apéndices, `#section.alphN.M.AlphK`)». Si se cambia, hay que ponerlo
en todas las copias. El libro copiará la suya cuando se avise.

El resto ya está hecho en el libro: enlaces a `github.io`,
`por_linea=2` y la copia de la coordinación.

**Respuesta desde NAcAL (2026-11-24).** Hecho, con el visto bueno de
Marcos: la fila del libro en la tabla de direcciones lleva la
redacción propuesta. Avisados los otros cuatro repositorios para que
copien el fichero.
