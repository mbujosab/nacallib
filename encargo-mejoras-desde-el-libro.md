
## Añadido el 2026-11-14 (tarde, 4): reglas de coordinación

Marcos ha acordado unas reglas para que el libro, NAcAL y el material
de clase sean coherentes entre sí. Están en
`coordinacion-entre-repositorios.md`, en la raíz de este repositorio
(el mismo texto está en los otros dos). Leerlo. En resumen: cada
sesión solo edita su repositorio; el libro manda en notación y
términos, NAcAL en lo que hace el código; el libro y NAcAL se miran el
uno al otro al tocar algo relacionado.

El prefacio y la introducción nuevos ya están en el libro (commit
`aa768d3`), con el Ejemplo 4 de Cayley-Hamilton.

## Añadido el 2026-11-15: la 0.4.1 está instalada

- Instalada desde la sesión del libro, a petición de Marcos, con las
  indicaciones de `empaquetadoPIP/empaquetadoPIP_nacal.org`: paquete
  construido desde `master` (commit `1a76c40`, que incluye `__all__`)
  e instalado con
  `pip install --user --no-deps --break-system-packages`.
- Antes de instalar se ejecutó todo el código Python del libro con la
  0.4.1 y se comparó con las salidas de la 0.3.0: las 335 salidas que
  guarda PythonTeX son idénticas y no hay errores. `from nacal import
  *` con `__all__` no rompe nada en el libro.
- No se ha comprobado el material de clase (`MatematicasII-SRC`).

## Respuesta desde NAcAL (2026-11-16)

Tratado: leídas las reglas de coordinación y la nota de la instalación
de la 0.4.1. No pedía cambios. Después se han cerrado la 0.5.0 y la
0.5.1.
