# NAcAL (nacallib)

## Encargos de otras sesiones

Los encargos de otras sesiones llegan a la raíz en ficheros
`encargo-*.md` y `respuesta-*.md`. Un fichero de esos nuevo o con
cambios sin confirmar en git es un encargo sin tratar. Las reglas
completas están en `coordinacion-entre-repositorios.md`.

- Un hook (`.claude/settings.json`) lo comprueba cada vez que Marcos
  escribe y avisa con una línea «ENCARGOS SIN TRATAR». Si el aviso no
  aparece al empezar la sesión, haz la comprobación a mano:
  `git status --short -- 'encargo-*.md' 'respuesta-*.md'`.
- Si hay alguno, léelo, di en pantalla cuál es y de qué repositorio
  viene, resume lo que pide en pocas líneas y pregunta a Marcos qué
  hacer. No cambies nada hasta que responda. Si Marcos ha pedido otra
  cosa en ese mensaje, atiéndela y da el aviso al final.
- Cuando el encargo esté tratado, anota la respuesta al final del
  fichero y confírmalo en git. Mientras no se confirme, el aviso
  seguirá saliendo.
- Para pedir algo a otro repositorio, escribe allí un fichero con uno
  de esos nombres y no lo confirmes en git: lo confirma quien lo
  recibe.

## Documentación y PDF

Todo sale de `nacal_src.org` (código del módulo y documentación). Los
pasos para regenerar el código, poner al día los resultados de los
ejemplos y exportar `nacal_src.pdf` están en `herramientas/LEEME.md`.
Antes de editar `nacal_src.org`, comprueba que no existe
`.#nacal_src.org`: indica que Marcos lo tiene abierto en Emacs con
cambios sin guardar.
