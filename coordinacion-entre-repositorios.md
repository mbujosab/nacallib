# Coordinación entre el libro, NAcAL, el material de clase y los vídeos

Acordado con Marcos el 2026-11-14 y ampliado el 2026-11-15. Este
fichero está copiado, con el mismo texto, en la raíz de estos
repositorios:

- `CursoDeAlgebraLineal-SRC`: el libro.
- `nacallib`: el módulo NAcAL de Python y su documentación.
- `MatematicasII-SRC`: las transparencias, los problemas y los
  exámenes de la asignatura.
- `VideosMates2`: los vídeos, con sus guiones, subtítulos y fichas.

Se revisan en paralelo, en sesiones distintas que no comparten
memoria. Estas reglas sirven para que lo que dice uno sea coherente
con lo que dicen los otros. Si se cambia este fichero, hay que cambiar
todas las copias.

Habrá un quinto actor: el repositorio de los Notebooks de Jupyter
(`NAcAL-Jupyter-Notebooks-SRC`). Todavía no tiene sesión ni copia de
este fichero. Cuando la tenga, su papel será mirar a todos los demás.

## Regla 1: cada sesión solo edita su repositorio

Lo que una sesión vea mal en otro repositorio lo deja como sugerencia
escrita en aquel repositorio, en un fichero de encargo, y avisa a
Marcos. No lo corrige ella. Así dos sesiones no cambian el mismo
fichero sin saberlo.

Los ficheros de encargo y de respuesta se nombran por su origen, como
los ya existentes: `encargo-...-desde-el-libro.md`,
`respuesta-al-encargo-del-libro.md`. Son ficheros de trabajo. Cada
añadido lleva fecha y se pone al final.

## Regla 2: quién manda en cada cosa

Cuando dos repositorios no coinciden, esto decide cuál cambia:

- **El libro manda** en notación, términos, definiciones, enunciados y
  en el nombre y el orden de los procedimientos.
- **NAcAL manda** en lo que el código hace: nombres de las órdenes,
  argumentos y salidas. Si el libro describe una orden o una salida
  que no coincide con el código, cambia el libro. Si lo que dice el
  libro es mejor que lo que hace el código, se anota como petición a
  NAcAL.
- **El material de clase no manda sobre los otros dos.** Sigue al
  libro en notación y términos y a NAcAL en el código. Sí decide lo
  suyo: qué cabe en una hora de exposición, en qué orden se cuenta en
  el aula y qué ejercicios se proponen. Una diferencia de orden con el
  libro justificada por el tiempo de clase no es una incoherencia.
- **Los vídeos tampoco mandan sobre el libro ni sobre NAcAL.** Siguen
  al libro en notación, términos y numeración de secciones, y a NAcAL
  en el código. Deciden lo suyo: el guion, el ritmo y lo que se
  muestra. Un vídeo publicado no se puede cambiar; sí sus subtítulos y
  su ficha. Por eso, si el libro cambia algo que un vídeo publicado ya
  dice, hay que avisarlo antes de cambiar el libro.
- **Los Notebooks, cuando entren, no mandan sobre nadie.** Siguen al
  libro, a NAcAL, al material de clase y a los vídeos.
- **Ante la duda, decide Marcos.**

Una asimetría: el libro lo leen los alumnos y los vídeos enlazan a sus
secciones, así que cambia despacio. Si la discrepancia es entre el
libro y NAcAL y no hay error en el libro, se adapta NAcAL.

## Regla 3: quién mira a quién

- **El libro y NAcAL se miran el uno al otro.**
- **El material de clase mira a los dos**, y ellos no tienen el
  encargo de mirarlo a él.
- **Los vídeos miran al libro y a NAcAL.** El libro, al revisar una
  lección, mira los guiones de los vídeos publicados de esa lección,
  porque ya no se pueden cambiar.
- **Los Notebooks, cuando entren, miran a todos los demás.**

Pero la información vuelve en los dos sentidos. Si al revisar una
clase o al preparar un vídeo aparece un error del libro o de NAcAL, o
una explicación que funciona mejor que la del libro, se anota:

- para el libro, en
  `CursoDeAlgebraLineal-SRC/PendientesDesdeLasClases.org` si viene de
  las clases y en
  `CursoDeAlgebraLineal-SRC/PendientesDesdeLosVideos.org` si viene de
  los vídeos;
- para NAcAL, en un fichero de encargo en `nacallib`.

## Cuándo mirar

No se hace una revisión cruzada general. La comprobación va unida al
trabajo que ya se hace:

- **Desde el libro:** al revisar una lección, mirar los apartados de
  la documentación de NAcAL que enlazan con ella. Las anclas
  `\hypertarget{target:...}` del libro dicen cuáles son.
- **Desde NAcAL:** al documentar o cambiar una función, mirar la
  sección del libro con la que enlaza.
- **Desde el material de clase:** al revisar una lección, comprobar
  sus transparencias, problemas y exámenes contra la lección del libro
  y contra la versión instalada de NAcAL.
- **Desde los vídeos:** al escribir un guion, comprobarlo contra la
  sección del libro que desarrolla y contra la versión de NAcAL con la
  que se genera el vídeo.

## Comprobaciones que no dependen de la memoria de nadie

- `nacallib/tests/test_enlaces_al_libro.py` comprueba que toda ancla
  del libro que usa la documentación de NAcAL existe.
- **Las anclas con nombre del libro no se renombran ni se quitan.** Lo
  mismo vale para la numeración de las secciones, a la que enlazan los
  vídeos.
- **Antes de instalar una versión nueva de NAcAL**, ejecutar todo el
  código Python del libro con la versión instalada y con la nueva y
  comparar las salidas que guarda PythonTeX
  (`pythontex-files-libro/libro.pytxmcr`). Hecho así con la 0.3.0. El
  material de clase también inserta salidas de NAcAL y conviene
  comprobarlo igual.
- Los cambios de NAcAL que afecten a las salidas deben ser opcionales.
  La salida por defecto no cambia sin decidirlo con Marcos.
- La instalación de NAcAL en el equipo de Marcos la usan el libro y el
  material de clase. Se instala una versión cerrada, no una copia de
  trabajo.
- **PythonTeX guarda los resultados y no vuelve a ejecutar el código
  si su texto no ha cambiado.** Los guarda en la carpeta auxiliar
  `pythontex-files-<documento>`. Por eso, tras cambiar de versión de
  NAcAL, una compilación normal no comprueba nada: reutiliza las
  salidas de la versión anterior. Para comprobar un documento con una
  versión nueva hay que forzar la ejecución, con
  `pythontex --rerun always --interpreter python:python3 <documento>.tex`
  o borrando antes esa carpeta. Vale para el libro y para el material
  de clase. (Añadido el 2026-11-15.)
