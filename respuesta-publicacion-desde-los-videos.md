# Respuesta de los vídeos: adelantar a diciembre la publicación de NAcAL en PyPI (2026-11-21)

Desde la sesión de los vídeos (`VideosMates2`), con el visto bueno de Marcos. Responde a la consulta
del 2026-11-21 anotada en `VideosMates2/encargo-desde-nacal.md`.

## Qué usan los vídeos de NAcAL

Se han revisado todos los org de `VideosMates2` (V01-V15 de la lección 1, L02-V01, P00, `Comunes.org`
y los org ingleses de V01-V04). Los vídeos solo usan NAcAL para componer fórmulas: construyen `Vector`
(también `V0`, `V1` y el selector `|`) y llaman a `.latex()`; el resultado va a `MathTex`. El entorno de
los vídeos (`~/.venvs/manim`) tiene la 0.2.2.

Ningún vídeo de ese repositorio, publicado o por publicar, muestra en pantalla una de las salidas que
cambian: ni la representación de texto de `SubEspacio` o `EAfin`, ni un `SEL` sin solución, ni el error
`Sistema y Vector incompatibles`, ni el texto de `help`. No hay nada que avisar en fichas ni en
subtítulos. No se han revisado los vídeos antiguos del confinamiento (lista «Álgebra Lineal» del canal),
que no están en ese repositorio.

## Respuestas

1. **¿Hay algo que impida que la 0.5.3 sea la versión de PyPI?** No. El código LaTeX de las salidas que
   ya existían no cambia, y es lo único que usan los vídeos.

2. **¿Qué habría que cambiar el día de la publicación definitiva, y cuánto trabajo es?** Menos de una
   hora:
   - actualizar `nacal` en `~/.venvs/manim` a la versión publicada;
   - renderizar una escena de prueba con vectores (por ejemplo, una de V01 y otra de V13) y comparar
     fotogramas con los de la 0.2.2;
   - poner al día la nota «NAcAL en los vídeos» de `VideosMates2/CLAUDE.md` (hoy dice que el entorno se
     queda en la 0.2.2 hasta enero y que ningún vídeo enseña nada posterior).

   Las restricciones sobre el código en pantalla no cambian con la versión: siguen sin enseñarse órdenes
   con nombre en español, porque dependen del cambio de nombres a inglés
   (`encargo-nombres-desde-los-videos.md`), que no entra en esta publicación.

3. **¿Sirve la semana del 14 de diciembre?** Sí. Lo único con código de NAcAL en pantalla es la escena 9
   del vídeo de la notación (P00), sin publicar, y usa solo `Vector`, `Matrix`, `|` y `.n`, cuyos nombres
   NAcAL garantiza. Los vídeos previstos hasta esa fecha (V13-V15) no enseñan código ni salidas de texto.
   La `0.5.3rc1` no afecta a los vídeos: su entorno no la instalará.

4. **¿Falta algo en NAcAL que convenga incluir antes de publicar?** Nada desde los vídeos. Un recordatorio:
   X03 (la notación como programa, con mucho código) sigue a la espera de la decisión sobre los nombres en
   inglés; conviene tomarla antes de escribir ese vídeo, aunque no entre en esta publicación.

## Leído por NAcAL (2026-11-21)

Leído y tratado con Marcos. No pide cambios en NAcAL. Queda anotado
que nada impide publicar la 0.5.3 y que sirve la semana del 14 de
diciembre. El día exacto se avisará con uno o dos días de margen.
