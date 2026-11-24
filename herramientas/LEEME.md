# Herramientas para mantener la documentación de NAcAL

Todo sale de `nacal_src.org`: el código del módulo (`nacal/nacal.py` y
`nacal/__init__.py`) y la documentación (`nacal_src.pdf`). Estos
guiones hacen sin abrir Emacs los dos pasos que más se repiten. Se
pueden lanzar desde cualquier carpeta.

## Orden habitual tras cambiar `nacal_src.org`

1. Guardar el fichero en Emacs (los guiones leen lo que hay en disco).
2. Si ha cambiado código del módulo, generarlo de nuevo:
   `emacs --batch -Q nacal_src.org -f org-babel-tangle`
   (en Emacs, `C-c C-v t`).
3. Si ha cambiado algún ejemplo, o código que afecte a su salida:
   `python3 herramientas/actualiza_resultados.py`
4. Pasar las pruebas: `python3 -m pytest tests -q`
5. Exportar el PDF: `sh herramientas/exporta_pdf.sh`
6. En Emacs, `M-x revert-buffer` si el paso 3 ha cambiado el fichero.

## `actualiza_resultados.py`

Vuelve a ejecutar, en el orden del documento, todos los ejemplos
(bloques `jupyter-python` que se exportan) y reescribe solo los
`#+RESULTS:` que no coinciden con la salida real.

    python3 herramientas/actualiza_resultados.py            # reescribe
    python3 herramientas/actualiza_resultados.py --mirar    # solo dice cuáles cambiarían

- No necesita Emacs ni Jupyter. Usa el módulo de este repositorio, no
  el instalado.
- Si `nacal_src.org` está abierto en Emacs con cambios sin guardar, se
  niega a escribir.
- Un resultado guardado en un cajón `:RESULTS:` no lo toca: avisa y
  hay que corregirlo a mano.
- Usa la misma maquinaria que `tests/test_documentacion.py`, que es la
  prueba que falla cuando un resultado guardado no es la salida real.

## `exporta_pdf.sh`

Exporta `nacal_src.org` a `nacal_src.pdf` con la configuración de
Emacs del usuario (`~/.emacs.d`).

    sh herramientas/exporta_pdf.sh

- Ejecuta antes el bloque `NombreEnChunksDeCodigo` del documento, que
  pone el nombre de cada trozo de código sobre su recuadro y parte las
  líneas largas de código.
- Al acabar dice si ha habido errores de LaTeX, referencias sin
  definir o líneas que se salen del margen más de 5pt, y borra los
  ficheros auxiliares. Devuelve un código distinto de cero si algo de
  eso falla.
- No vuelve a calcular los ejemplos: exporta los resultados guardados.
- Al subir a `master` un `nacal_src.pdf` nuevo, la página web
  (`https://mbujosab.github.io/nacallib/`) se publica sola.

## Lo mismo desde Emacs

- Al abrir `nacal_src.org`, Emacs ejecuta dos bloques (lo piden las
  variables locales del final del fichero): `NombreEnChunksDeCodigo` y
  `displayMathOutput`. La primera vez pregunta; con `!` no vuelve a
  preguntar.
- Un ejemplo: cursor dentro del bloque y `C-c C-c`.
- Los ejemplos de una sección: `C-c C-v s`.
- Todos los ejemplos: `C-c C-v b` (los trozos del módulo no se
  ejecutan, llevan `:eval no`).
- Exportar: `C-c C-e l p`.

## Si un ejemplo no cabe en el ancho de la página

`desarrollo(objeto, por_linea=N)` reparte un proceso de eliminación en
líneas de `N` matrices. Si aun así no cabe, se puede reducir la letra
de ese ejemplo con una línea `#+LATEX: {\scriptsize` antes del bloque
y otra `#+LATEX: }` después del resultado (véase el ejemplo de
combinaciones lineales de matrices).
