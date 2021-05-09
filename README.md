[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbujosab/nacallib/master?filepath=doc%2FNotebooks%2FNotebook.ipynb)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbujosab/nacal-Jupyter-Notebooks/master)

# Paquete "nacal" (Notación Asociativa para un Curso de Álgebra Lineal)

Este paquete implementa la notación empleada en mi curso de [Álgebra Lineal (Matemáticas II)](https://www.ucm.es/fundamentos-analisis-economico2/algebra-2)

Aunque no es estrictamente necesario, está pensado para su uso con los [Notebooks de Jupyter](https://jupyter.org/).
En dicho entorno la librería muestra como llegar a la mayoría de los resultados empleando el método de 
eliminación. Es decir, este paquete o librería, no solo resuelve sistemas de ecuaciones, invierte 
matrices, calcula determinantes, diagonaliza matrices tanto por semejanza como por congruencia, etc. 
Sino que muestra los pasos empleados para llegar al resultado como si se hiciera con lápiz y papel. 
Además, también permite trabajar con subespacios y espacios afines de
![equation](https://latex.codecogs.com/gif.latex?%5Cmathbb%7BR%7D%5En). También puede trabajar de 
manera simbólica, pues emplea los objetos básicos de la librería [Sympy](https://www.sympy.org/en/index.html).

La documentación explica la programación del código y sirve como
material adicional al [libro del curso](https://github.com/mbujosab/CursoDeAlgebraLineal)
(este módulo es una implementación literal de lo mostrado en
dicho libro). Puede ver el uso de la librería sin necesidad de
instalar nada accediendo a los Notebooks de Jupyter alojados en
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mbujosab/nacallib/master?filepath=doc%2FNotebooks%2FNotebook.ipynb)
con su navegador de páginas web (puede tardar unos minutos en
cargar la librería y el Notebook de demostración).

## Instalación
[nacal](https://pypi.org/project/nacal/) funciona con Python 3.7. Puede instalar el paquete desde PyPI via pip:

```sh
pip3 install nacal
```

[nacal](https://pypi.org/project/nacal/) emplea [Sympy](https://www.sympy.org/en/index.html). Para instalar Sympy:
```sh
pip3 install sympy
```

## Uso
Para emplear esta librería en una consola de Python, una vez instalada:
```sh
pyhton3
>>> from nacal import *
```

Para emplearla en un Notebook de Jupyter, ejecute en un "Cell" de código
```
from nacal import *
```


## Desinstalación
Para desinstalar [nacal](https://pypi.org/project/nacal/):

```sh
pip3 uninstall nacal
```
