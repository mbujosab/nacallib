# Encargo desde los Notebooks: el mensaje «Sistema y Vector incompatibles»

Viene de la sesión de los Notebooks (2026-11-18). Sale de la auditoría
independiente de los Notebooks de la Lección 2
(`../NAcAL-Jupyter-Notebooks-SRC/informe-auditoria-leccion2.md`,
hallazgo 4 y apartado 6). No es urgente. Los Notebooks ya explican el
mensaje en el texto.

## Qué se ha visto

`Vector([1,2,3]) * Vector([1,2])` y `A * Vector([1,2])`, con `A` de
tres columnas, dan `ValueError: Sistema y Vector incompatibles`.
Probado con la 0.2.2 y con el código fuente de la 0.5.2. Es el primer
error que ve el alumno en la Lección 2. Los tres Notebooks de la
lección le piden provocarlo y leerlo.

1. El mensaje no dice cuántas componentes tiene cada operando. Con una
   matriz cuadrada el alumno no sabe si el vector debe tener tantas
   componentes como columnas o como filas.
2. Con `idioma('en')` el mensaje sigue saliendo en español. Los
   Notebooks en inglés avisan de ello en el texto.
3. «Sistema» aparece en el mensaje aunque el alumno haya escrito dos
   vectores o una matriz y un vector.

## Qué se pide

Valorar un mensaje que diga los números, por ejemplo «no se pueden
multiplicar: el primero tiene 3 componentes (o columnas) y el segundo
tiene 2», y que se traduzca con `idioma('en')`.

Los Notebooks citan hoy el mensaje literal, `ValueError: Sistema y
Vector incompatibles`, en `Lecc02-Secc2.1` y `Lecc02-Secc2.1-Eng`. Si
el mensaje cambia, hay que avisar a los Notebooks para cambiar esa
frase cuando la versión nueva llegue a PyPI.

## Sin petición

`1|True` y `2|3` no dan error, porque Python los calcula antes de que
NAcAL intervenga. No tiene arreglo en NAcAL. La función `dibuja` del
Notebook `DibujarVectores` comprueba ahora sus argumentos por ese
motivo.

## Respuesta desde NAcAL (2026-11-21)

Tratado en la 0.5.3 (sin publicar en PyPI; prevista para la primera
quincena de enero de 2027).

- El mensaje dice qué se ha multiplicado y los dos números que no
  coinciden. `Vector([1,2,3]) * Vector([1,2])` da `ValueError: No se
  pueden multiplicar: el vector de la izquierda tiene 3 componentes y
  el vector de la derecha tiene 2 componentes`. Con una matriz de tres
  columnas a la izquierda: `la matriz de la izquierda tiene 3 columnas
  y el vector de la derecha tiene 2 componentes`. Con una matriz a la
  derecha se dan sus filas. «Sistema» solo aparece si el operando de la
  izquierda no es ni un `Vector` ni una `Matrix`.
- Con `idioma('en')` sale en inglés: `Cannot multiply: the matrix on
  the left has 3 columns and the vector on the right has 2 components`.
  Es el único mensaje de error que se traduce; los demás siguen en
  español (decidido con Marcos).
- El mensaje «Sistema y Matrix incompatibles» cambia de la misma forma.
- Banco de material, 0.5.2 frente a 0.5.3: el libro, las clases y los
  exámenes no cambian (89 casos iguales). De los 27 Notebooks cambian
  cuatro, y solo en ese mensaje: `Lecc02-Secc2.1`, `Lecc02-Secc2.2` y
  sus versiones en inglés.
- Aviso a los Notebooks anotado en
  `../NAcAL-Jupyter-Notebooks-SRC/encargo-desde-nacal.md`.
