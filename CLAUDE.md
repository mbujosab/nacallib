# NAcAL (nacallib)

## Encargos de otras sesiones

Los encargos de otras sesiones llegan a la carpeta
`../coordinacion/para-nacal/`, en ficheros `encargo-*.md` y
`respuesta-*.md`. Esa carpeta tiene su propio repositorio git, que no
se publica (desde el 2026-11-27 los encargos no están en `nacallib`,
que es público). Un fichero nuevo o con cambios sin commit en esa
carpeta es un encargo sin tratar. Las reglas completas están en
`../coordinacion/coordinacion-entre-repositorios.md` (única copia desde
el 2026-11-27; ya no hay una en `nacallib`) y en
`../coordinacion/LEEME.md`. Si se acuerda con Marcos un cambio de esas
reglas, se escribe allí, se hace commit solo de ese fichero y se avisa
a las demás sesiones con un encargo.

- Un hook (`.claude/settings.json`) lo comprueba cada vez que Marcos
  escribe y avisa con una línea «ENCARGOS SIN TRATAR». Si el aviso no
  aparece al empezar la sesión, haz la comprobación a mano:
  `git -C ../coordinacion status --short -- para-nacal/`.
- Si hay alguno, léelo, di en pantalla cuál es y de qué repositorio
  viene, resume lo que pide en pocas líneas y pregunta a Marcos qué
  hacer. No cambies nada hasta que responda. Si Marcos ha pedido otra
  cosa en ese mensaje, atiéndela y da el aviso al final.
- Cuando el encargo esté tratado, anota la respuesta al final del
  fichero y haz commit en el repositorio de coordinación, solo de
  esta subcarpeta: `git -C ../coordinacion add para-nacal/` y después
  el commit. Nunca `git add -A` allí. Mientras no haya commit, el
  aviso seguirá saliendo.
- Para pedir algo a otra sesión, escribe un fichero con uno de esos
  nombres en su subcarpeta (`para-el-libro/`, `para-las-clases/`,
  `para-los-notebooks/`, `para-los-videos/`, `para-nacaltex/`,
  `para-la-pagina-personal/`) y no
  hagas commit: lo hace quien lo recibe.
- Mientras las otras sesiones no hayan cambiado al sitio nuevo, el
  hook mira también la raíz de `nacallib`. Un encargo que llegue ahí
  se trata igual y se mueve a `../coordinacion/para-nacal/`. No se
  hace commit de él en `nacallib`.
- En `nacallib`, haz `git add` nombrando los ficheros, no con `-A`.

## Documentación y PDF

Todo sale de `nacal_src.org` (código del módulo y documentación). Los
pasos para regenerar el código, poner al día los resultados de los
ejemplos y exportar `nacal_src.pdf` están en `herramientas/LEEME.md`.
Antes de editar `nacal_src.org`, comprueba que no existe
`.#nacal_src.org`: indica que Marcos lo tiene abierto en Emacs con
cambios sin guardar.
