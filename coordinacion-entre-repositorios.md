# Coordinación entre el libro, NAcAL y el material de clase

Acordado con Marcos el 2026-11-14. Este fichero está copiado, con el
mismo texto, en la raíz de los tres repositorios:

- `CursoDeAlgebraLineal-SRC`: el libro.
- `nacallib`: el módulo NAcAL de Python y su documentación.
- `MatematicasII-SRC`: las transparencias, los problemas y los
  exámenes de la asignatura.

Los tres se revisan en paralelo, en sesiones distintas que no
comparten memoria. Estas reglas sirven para que lo que dice uno sea
coherente con lo que dicen los otros. Si se cambia este fichero, hay
que cambiar las tres copias.

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
- **Ante la duda, decide Marcos.**

Una asimetría: el libro lo leen los alumnos y los vídeos enlazan a sus
secciones, así que cambia despacio. Si la discrepancia es entre el
libro y NAcAL y no hay error en el libro, se adapta NAcAL.

## Regla 3: quién mira a quién

- **El libro y NAcAL se miran el uno al otro.**
- **El material de clase mira a los dos**, y ellos no tienen el
  encargo de mirarlo a él.

Pero la información vuelve en los dos sentidos. Si al revisar una
clase aparece un error del libro o de NAcAL, o una explicación de
clase que funciona mejor que la del libro, se anota:

- para el libro, en
  `CursoDeAlgebraLineal-SRC/PendientesDesdeLasClases.org`;
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
