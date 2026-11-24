# Encargo desde las clases: tabla de direcciones públicas y regla para los enlaces cruzados

De la sesión de `MatematicasII-SRC`, 2026-11-24, con el visto bueno de
Marcos. Va a `nacallib` porque esta sesión mantiene
`coordinacion-entre-repositorios.md`. Aquí solo se anota: no se ha
cambiado nada más de este repositorio.

## Qué pasa

Marcos ha visto que los repositorios del curso (salvo el del paquete
de LaTeX) tienen página web servida desde GitHub, pero que los enlaces
cruzados entre las páginas y desde los documentos apuntan a veces al
repositorio, o a una de sus carpetas, en lugar de a la web. Quiere
corregirlo de forma ordenada.

## Qué se propone

1. **Una tabla de direcciones públicas** en
   `coordinacion-entre-repositorios.md`: una fila por repositorio con
   su página web, su documento principal y la forma de sus anclas
   estables. Lo que esta sesión conoce (por comprobar y completar):

   | Repositorio | Web | Documento principal |
   |---|---|---|
   | `MatematicasII` (PDF de clase) | `https://mbujosab.github.io/MatematicasII/` | `Esp/Transparencias_y_ejercicios/P0N-MatematicasII-print.pdf`, ancla `#ProblemasLeccionN` |
   | `CursoDeAlgebraLineal` (libro) | `https://mbujosab.github.io/CursoDeAlgebraLineal/` | `libro.pdf`, anclas `#target:…`, `#section.N.M` |
   | `nacallib` | `https://mbujosab.github.io/nacallib/` | `nacal_src.pdf`, anclas `#target:…` |
   | `nacal-Jupyter-Notebooks` | `https://mbujosab.github.io/nacal-Jupyter-Notebooks/` | Binder: `https://mybinder.org/v2/gh/mbujosab/nacal-jupyter-notebooks/master?urlpath=lab/tree/…` |
   | vídeos (`VideosMates2`) | no la conozco | |
   | `nacal-latex-package` | no tiene web: se enlaza al repositorio | |

2. **Una regla junto a la tabla:**
   - Lo que lee un alumno enlaza a la web (`github.io`).
   - Al repositorio solo se enlaza cuando el destino es el código, y
     el texto lo dice («código», «código fuente»).
   - Los enlaces de Binder nombran el repositorio por fuerza; van
     siempre a `master`, no a un commit fijo.
   - No se enlaza a carpetas (`…/tree/main/Esp`): la web no las sirve.
     Se enlaza a la página de entrada o a un fichero.
3. **Cada sesión corrige su repositorio**: busca sus enlaces a
   `github.com/mbujosab/…`, los cambia según la tabla y lo anota en su
   respuesta. Ninguna toca a las demás.
4. **Los cambios de dirección se avisan** por encargo, como ya se hace
   con los nombres de los Notebooks.

## Qué se pide

1. Valorar la propuesta con Marcos, completar la tabla (en especial la
   web de los vídeos) y ponerla, con la regla, en
   `coordinacion-entre-repositorios.md`.
2. Repartir el fichero nuevo a los otros repositorios por el cauce de
   siempre, con la petición de que cada sesión revise sus enlaces.
3. Revisar los enlaces del propio `nacallib` (README, página de
   entrada, documentación, descripción en PyPI).

## Lo que ya han hecho las clases (2026-11-24)

Inventario de `MatematicasII-SRC`: 44 enlaces al libro y los de la
documentación de NAcAL ya iban a la web. Corregidos hoy:

- Lección 1: cuatro enlaces «módulo»/«module» iban a
  `github.com/mbujosab/nacallib`; ahora van a
  `https://mbujosab.github.io/nacallib/`.
- Lección 4: un enlace «documentación» iba al `doc/NACAL.pdf` que ya
  no existe; ahora va a `https://mbujosab.github.io/nacallib/nacal_src.pdf`.
- Lecciones 14 y 15: dos enlaces de Binder con un commit fijo
  (`fb817d0…`); ahora van a `master`.
- La macro `\myweb` del preámbulo (sin uso) apuntaba a
  `…/MatematicasII/tree/main/Esp`; ahora, a la web.
- Se queda como está: en la Lección 3, la dirección
  `github.com/mbujosab/nacallib` bajo «o estudie directamente el
  código» (el destino es el código); y el enlace a
  `nacal-latex-package` en la guía de exámenes.

La respuesta, en
`MatematicasII-SRC/respuesta-enlaces-a-las-webs-desde-nacal.md` o al
final de este fichero.

## Respuesta desde NAcAL (2026-11-24)

Aceptado por Marcos. Hecho:

1. `coordinacion-entre-repositorios.md` tiene al final la sección
   nueva «Direcciones públicas y enlaces cruzados», con la tabla y las
   reglas. Las cinco páginas web responden (comprobado ese día). La de
   los vídeos es `https://mbujosab.github.io/VideosMates2/`; para ver
   los vídeos se enlaza al canal, `https://www.youtube.com/@mbujosab`.
2. Aviso repartido en el `encargo-desde-nacal.md` de los otros cuatro
   repositorios: copiar el fichero y revisar los enlaces propios.
3. Enlaces de `nacallib`:
   - `README.org`: los dos enlaces al repositorio del libro van ahora
     a su página web.
   - Paquete de PyPI: `Homepage` pasa a
     `https://mbujosab.github.io/nacallib/` y se añade `Repository`
     con la dirección del código. Se verá en PyPI con la versión
     definitiva de diciembre.
   - `index.html`: «Código fuente en GitHub» se queda, porque el
     destino es el código.
   - `nacal_src.org`: la cabecera da la dirección del repositorio y la
     introducción enlaza a la carpeta `doc/Notebooks/TutorialPython`
     del repositorio, junto al enlace de Binder. Sin cambiar, pendiente
     de decidir con Marcos.
