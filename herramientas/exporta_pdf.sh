#!/bin/sh
# Exporta nacal_src.org a nacal_src.pdf sin abrir Emacs, con la configuración
# de Emacs del usuario (~/.emacs.d).
#
# Ejecuta antes el bloque NombreEnChunksDeCodigo, que pone el nombre de cada
# trozo de código sobre su recuadro y parte las líneas largas. Al acabar dice
# si ha habido errores, referencias sin definir o líneas que se salen del
# margen más de 5pt, y borra los ficheros auxiliares.
#
# Uso, desde cualquier carpeta:   sh herramientas/exporta_pdf.sh
#
# Los resultados de los ejemplos no se vuelven a calcular al exportar: para
# eso está herramientas/actualiza_resultados.py.

cd "$(dirname "$0")/.." || exit 1

if [ -L .#nacal_src.org ]; then
    echo "AVISO: nacal_src.org tiene cambios sin guardar en Emacs; se exporta lo que hay en disco"
fi

emacs --batch --init-directory ~/.emacs.d \
      -l ~/.emacs.d/early-init.el -l ~/.emacs.d/init.el \
      --eval '(setq enable-local-variables :all enable-local-eval t org-latex-remove-logfiles nil)' \
      nacal_src.org \
      --eval '(let ((org-confirm-babel-evaluate nil)) (org-babel-goto-named-src-block "NombreEnChunksDeCodigo") (org-babel-execute-src-block))' \
      -f org-latex-export-to-pdf 2>&1 | tail -2

estado=0
if [ ! -f nacal_src.log ]; then
    echo "No se ha generado nacal_src.log: la exportación ha fallado"
    exit 1
fi
grep 'Output written' nacal_src.log || { echo "No se ha escrito el PDF"; estado=1; }
if grep -q '^!' nacal_src.log; then
    echo "ERRORES de LaTeX:"; grep -n -A2 '^!' nacal_src.log | head -20; estado=1
fi
if grep -q 'undefined' nacal_src.log; then
    echo "Referencias sin definir:"; grep -n 'undefined' nacal_src.log | head; estado=1
fi
if ! grep -q 'label={' nacal_src.tex; then
    echo "Los trozos de código han salido sin su nombre"; estado=1
fi
desbordes=$(grep 'Overfull \\hbox' nacal_src.log |
            sed 's/.*(\([0-9.]*\)pt too wide).*/\1/' | awk '$1 > 5' | wc -l)
if [ "$desbordes" -gt 0 ]; then
    echo "Líneas que se salen del margen más de 5pt: $desbordes"
    grep 'Overfull \\hbox' nacal_src.log | awk -F'[(p]' '$2+0 > 5' | head
    estado=1
fi
[ "$estado" -eq 0 ] && echo "Sin errores, sin referencias rotas y nada se sale del margen"

rm -f nacal_src.log nacal_src.fdb_latexmk nacal_src.fls nacal_src.tex
exit $estado
