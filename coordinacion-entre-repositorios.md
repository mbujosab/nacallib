# Coordinación entre el libro, NAcAL, el material de clase, los vídeos y los Notebooks

Acordado con Marcos el 2026-11-14 y ampliado el 2026-11-15. Este
fichero está copiado, con el mismo texto, en la raíz de estos
repositorios:

- `CursoDeAlgebraLineal-SRC`: el libro.
- `nacallib`: el módulo NAcAL de Python y su documentación.
- `MatematicasII-SRC`: las transparencias, los problemas y los
  exámenes de la asignatura.
- `VideosMates2`: los vídeos, con sus guiones, subtítulos y fichas.
- `NAcAL-Jupyter-Notebooks-SRC`: los Notebooks de Jupyter de cada
  lección, en español y en inglés.

Se revisan en paralelo, en sesiones distintas que no comparten
memoria. Estas reglas sirven para que lo que dice uno sea coherente
con lo que dicen los otros. Si se cambia este fichero, hay que cambiar
todas las copias.

Los Notebooks son el quinto actor, desde el 2026-11-15. Su papel es
mirar a todos los demás.

El paquete de LaTeX `nacal.sty` es el sexto actor, desde el 2026-11-16.
Su repositorio está en
`~/SynologyDrive/Nextcloud/Docencia/Notacion/NAcALTex` y no tiene
copia de este fichero: tiene sus propias reglas en su `CLAUDE.md`
(banco de pruebas, versiones etiquetadas). El libro manda en qué
notación se usa; el paquete manda en cómo se compone. Las sugerencias
al paquete se le dejan en un fichero de encargo, como a los demás. El
paquete no cambia lo que imprime una macro existente sin acordarlo con
Marcos, porque todos los demás repositorios compilan con él.

## Regla 1: cada sesión solo edita su repositorio

Lo que una sesión vea mal en otro repositorio lo deja como sugerencia
escrita en aquel repositorio, en un fichero de encargo, y avisa a
Marcos. No lo corrige ella. Así dos sesiones no cambian el mismo
fichero sin saberlo.

Los ficheros de encargo y de respuesta se nombran por su origen, como
los ya existentes: `encargo-...-desde-el-libro.md`,
`respuesta-al-encargo-del-libro.md`. Son ficheros de trabajo. Cada
añadido lleva fecha y se pone al final.

**Aviso automático de encargos (desde el 2026-11-16).** Para que Marcos
no tenga que avisar en cada repositorio de que hay algo que leer:

- El nombre del fichero empieza siempre por `encargo-` o por
  `respuesta-`, está en la raíz del repositorio de destino y acaba en
  `.md`. (El libro recibe además los `Pendientes*.org` que ya existen.)
- Quien escribe un encargo en el repositorio de otro **no lo confirma
  en git**. Lo confirma quien lo recibe, cuando lo ha tratado y ha
  anotado su respuesta al final. Así, un fichero de encargo nuevo o con
  cambios sin confirmar es un encargo sin tratar.
- Cada repositorio tiene un hook en `.claude/settings.json` que ejecuta
  `git status --short -- 'encargo-*.md' 'respuesta-*.md'` cada vez que
  Marcos escribe, y una regla en su `CLAUDE.md`: si sale algo, la
  sesión lo lee, dice en pantalla qué es y de dónde viene, lo resume y
  pregunta a Marcos qué hacer. No cambia nada hasta que responda.
- Un encargo ya tratado que siga sin confirmar hace saltar el aviso en
  cada mensaje. Se arregla confirmándolo.

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
- **Los Notebooks no mandan sobre nadie.** Siguen al libro, a NAcAL,
  al material de clase y a los vídeos. Deciden lo suyo: qué ejemplos
  y ejercicios se proponen para trabajar con el ordenador.
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
- **Los Notebooks miran a todos los demás.**

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
- **Desde los Notebooks:** al revisar los de una lección,
  compararlos con la lección del libro, con sus vídeos y con la
  versión de NAcAL que reciben los alumnos.
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
- **El libro enlaza con Notebooks concretos.** Los enlaces están en la
  Lección 1 y apuntan a las carpetas `Esp/` y `Eng/` del repositorio
  público de los Notebooks. No se renombran ni se mueven Notebooks sin
  anotar antes el cambio para el libro. Ya ocurrió una vez y los
  enlaces del libro dejaron de funcionar. (Añadido el 2026-11-15.)
- **La versión de NAcAL que reciben los alumnos es la de PyPI**, que
  es la que instalan `pip install nacal` y mybinder. Puede ser más
  antigua que la instalada en el equipo de Marcos, con la que se
  compilan el libro y el material de clase. El 2026-11-22 PyPI tiene
  la 0.2.2 y el equipo de Marcos la 0.5.5. Lo que se enseñe a los
  alumnos tiene que funcionar con la versión de PyPI. (Añadido el
  2026-11-15; versiones puestas al día el 2026-11-22.)
- **Calendario para que la versión de PyPI y la del equipo de Marcos
  coincidan al empezar el curso** (finales de enero de 2027). Acordado
  con Marcos el 2026-11-15 y adelantado el 2026-11-21, tras consultar
  a los demás repositorios, porque nadie usa el módulo este
  cuatrimestre:
  1. Noviembre: hecho. NAcAL 0.5.5 está cerrada e instalada en el
     equipo de Marcos desde el 2026-11-21. PyPI tiene además una
     versión de prueba, la `0.5.5rc1`. No la recibe `pip install
     nacal`, que sigue instalando la 0.2.2; se instala pidiéndola por
     su número, `pip install nacal==0.5.5rc1`. Sirve para probar
     mybinder antes de la definitiva.
  2. Hasta mediados de diciembre: el libro y el material de clase se
     compilan con la 0.5.5; lo que falle se anota como encargo en
     `nacallib`. El material de clase ya lo ha hecho (2026-11-22), sin
     errores.
  3. Semana del 14 de diciembre de 2026: Marcos publica en PyPI la
     versión definitiva, la 0.5.5 o la que la corrija. El día exacto
     se avisa a los Notebooks con uno o dos días de margen, porque
     tienen que publicar sus cambios el mismo día y después de NAcAL
     (mybinder no instala la versión nueva hasta que su repositorio
     público tiene un commit nuevo). Durante el curso no se cambia la
     de PyPI salvo fallo grave.

  Hasta que se publique la definitiva, lo añadido después de la 0.2.2
  (`desarrollo`, `idioma`, `Polinomio`, `fracciones`,
  `es_antisimetrica`, argumentos nuevos) puede usarse en lo que
  compila Marcos (libro, transparencias, exámenes), pero no en código
  que ejecuten los alumnos. La versión de prueba no cambia esto: los
  alumnos reciben la definitiva. (Añadido el 2026-11-15; puesto al día
  el 2026-11-22.)
