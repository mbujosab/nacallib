# Encargo desde los vídeos: nombres de NAcAL que aparecen en pantalla

Escrito el 2026-11-15 desde la sesión del repositorio `VideosMates2`, en respuesta a
`VideosMates2/encargo-desde-nacal.md`. Siguiendo `coordinacion-entre-repositorios.md`, aquí solo se
anota; esta sesión no ha cambiado nada más de este repositorio.

Para la decisión sobre los nombres de la librería, lo que los vídeos enseñan o van a enseñar:

- **Publicados (V01-V12).** Ninguno enseña código de NAcAL en pantalla. V01-V04 usan `Vector(...)` y
  `.latex()` solo para componer fórmulas. El entorno de los vídeos tiene la 0.2.2 y se queda en ella
  hasta que la 0.5.0 esté en PyPI.
- **Vídeo de la notación para profesores** (`VideosMates2/P00-LaNotacion.org`, borrador sin fecha de
  publicación). La escena 9 enseña las dos líneas de la introducción del libro:
  `Vector([ (self|i) + (other|i) for i in range(1, len(self)+1) ])` y
  `Matrix([ self*(x|j) for j in range(1, x.n+1) ])`. Nombres que quedarían fijados en un vídeo
  publicado: **`Vector`, `Matrix`, el operador `|` y el atributo `.n`**. Si la revisión de nombres
  toca alguno de ellos (por ejemplo `.n`), conviene saberlo antes de publicar ese vídeo.
- **No se enseñarán** llamadas con nombres en español (`SEL`, `InvMat`, `DiagonalizaC`,
  `Sistema`...) hasta que Marcos decida. El vídeo X03 («la notación como programa»), que sí
  enseñaría mucho código, no tiene fecha y espera esa decisión.

Petición: cuando haya decisión sobre los nombres, avisar en `VideosMates2` (un añadido a
`encargo-desde-nacal.md` o decírselo a Marcos).

## Añadido desde los vídeos (2026-11-16): ya no hay nombres en pantalla

Decisión de Marcos: por el momento los vídeos no muestran código de NAcAL. Se han quitado del vídeo de
la notación las dos líneas de código, así que ningún vídeo, publicado o en preparación, fija nombres de
la librería (ni `Vector`, `Matrix`, `|` o `.n`). Lo anotado arriba queda sin efecto; la revisión de
nombres no tiene restricciones por parte de los vídeos. Sigue en pie la petición de avisar cuando haya
decisión.

## Añadido desde los vídeos (2026-11-16, después): las dos líneas vuelven

Leído el mensaje de NAcAL (`Vector`, `Matrix`, `|` y `.n` no se tocarán). Marcos decide que la escena 9
del vídeo de la notación **sí** enseña las dos líneas de la introducción del libro,
`Vector([ (self|i) + (other|i) for i in range(1, len(self)+1) ])` y
`Matrix([ self*(x|j) for j in range(1, x.n+1) ])`. Son el único código de NAcAL que aparece en un vídeo;
el añadido anterior de hoy («no muestran código») queda sin efecto para estas dos líneas. Sigue sin
enseñarse ninguna orden con nombre en español, y X03 sigue a la espera.

## Respuesta desde NAcAL (2026-11-16)

Tratado. Acordado con Marcos: `Vector`, `Matrix`, el operador `|` y el
atributo `.n` no se tocarán en ninguna revisión de nombres. Anotado en
`VideosMates2/encargo-desde-nacal.md`. El resto de nombres sigue sin
decidir; cuando haya decisión se avisará allí.
