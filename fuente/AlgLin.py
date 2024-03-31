# coding=utf8
import sympy
from IPython.display import display, Math, display_png
import tempfile
from os.path import join           

NumberTypes = (int, float, complex, sympy.Basic)
es_numero   = lambda x: isinstance(x, NumberTypes) and not isinstance(x, bool)

def fracc(a,b):
    """Transforma la fracción a/b en un número racional si ello es posible"""
    if all( [ isinstance(i, (int, float, sympy.Rational) ) for i in (a,b) ] ):
        return sympy.Rational(a, b)
    else:
        return a/b
    
def numer(a,b):
    """Devuelve el numerador de a/b si la fracción es un número racional,
       si no devuelve a/b"""
    if all( [ isinstance(i, (int, float, sympy.Rational) ) for i in (a,b) ] ):
        return fracc(a,b).p
    else:
        return a/b 

def denom(a,b):
    """Devuelve el denominador de a/b si la fracción es un número
       racional, si no devuelve 1"""
    if all( [ isinstance(i, (int, float, sympy.Rational) ) for i in (a,b) ] ):
        return fracc(a,b).q
    else:
        return 1


def html(TeX):
    """ Plantilla HTML para insertar comandos LaTeX """
    return "<p style=\"text-align:center;\">$" + TeX + "$</p>"
    

def latex(a):
    """Método latex general"""
    try:
        return a.latex()
    except:
        return sympy.latex(a)
    

def pinta(data):
    """Muestra en Jupyter la representación latex de data"""
    display(Math(latex(data)))


def CreaLista(t):
    """Devuelve t si t es una lista; si no devuelve la lista [t]"""
    return t if isinstance(t, list) else [t]


def CreaSistema(t):
    """Devuelve t si t es un Sistema; si no devuelve un Sistema que contiene t"""
    return t if isinstance(t, Sistema) else Sistema(CreaLista(t))


def primer_no_nulo(s):
    """Primer elemento no nulo en un sistema de sistemas de números"""
    c = [] if es_numero(s) else s.primer_no_nulo()
    return c + primer_no_nulo(CreaSistema(s)|c[0]) if c!=[] else []

def ultimo_no_nulo(s):
    """Primer elemento no nulo en un sistema de sistemas de números"""
    c = [] if es_numero(s) else s.ultimo_no_nulo()
    return c + ultimo_no_nulo(CreaSistema(s)|c[0]) if c!=[] else []

def elementoPivote(s):
    """Primer elemento no nulo"""
    if es_numero(s):
        return s
    elif CreaSistema(s).elementoPivote():
        return elementoPivote(CreaSistema(s).extractor(CreaSistema(s).primer_no_nulo()))
    else:
        None

def elementoAntiPivote(s):
    """Último elemento no nulo"""
    if es_numero(s):
        return s
    elif CreaSistema(s).elementoAntiPivote():
        return elementoAntiPivote(CreaSistema(s).extractor(CreaSistema(s).ultimo_no_nulo()))
    else:
        None


def particion(s,n):
    """ genera la lista de particionamiento a partir de un conjunto y un número
    >>> particion({1,3,5},7)

    [[1], [2, 3], [4, 5], [6, 7]]
    """
    s = {e for e in s if e<=n}
    p = sorted(list(s | set([0,n])))
    return [ list(range(p[k]+1,p[k+1]+1)) for k in range(len(p)-1) ]
    

def filtradopasos(pasos):
    abv = pasos.abreviaturas if isinstance(pasos,T) else pasos
           
    p = [T([j for j in T([abv[i]]).abreviaturas if (isinstance(j,set) and len(j)>1)\
               or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)       \
               or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])    \
                                             for i in range(0,len(abv)) ]

    abv = [ t for t in p if t.abreviaturas] # quitamos abreviaturas vacías de la lista
    
    return T(abv) if isinstance(pasos,T) else abv


def simplify(self):
    """Devuelve las expresiones simplificadas"""
    if isinstance(self, (list, tuple)):
        return type(self)([ simplify(e) for e in self ])
    elif isinstance(self, Sistema):
        self.lista=[ simplify(e) for e in self ]
        return self
    elif isinstance(self, T):
        self.abreviaturas=[ simplify(op) for op in self.abreviaturas ]
        return self
    else:
        return (sympy.sympify(self)).simplify()
    
def factor(self):
    """Devuelve las expresiones factorizadas"""
    if isinstance(self, (list, tuple)):
        return type(self)([ factor(e) for e in self ])
    elif isinstance(self, Sistema):
        self.lista=[ factor(e) for e in self ]
        return self
    elif isinstance(self, T):
        self.abreviaturas=[ factor(op) for op in self.abreviaturas ]
        return self
    else:
        return sympy.factor(self)

def expand(self):
    """Devuelve las expresiones factorizadas"""
    if isinstance(self, (list, tuple)):
        return type(self)([ expand(e) for e in self ])
    elif isinstance(self, Sistema):
        self.lista=[ expand(e) for e in self ]
        return self
    elif isinstance(self, T):
        self.abreviaturas=[ expand(op) for op in self.abreviaturas ]
        return self
    else:
        return sympy.expand(self)


def rprElim(data, pasos, TexPasosPrev=[], sust=[]):
    """Escribe en LaTeX los pasos efectivos y los sucesivos sistemas"""
    A     = data.fullcopy().subs(sust)
    tex   = latex(A) if not TexPasosPrev else TexPasosPrev
    
    # transformaciones por la izquierda
    for  _,pasoDeEliminacion in enumerate(pasos[0][::-1]):
        if data.es_arreglo_rectangular(): # entonces transforman las filas
            tex += '\\xrightarrow[' + latex( pasoDeEliminacion.subs(sust) ) + ']{}' 
            tex += latex( factor((pasoDeEliminacion & A).subs(sust)) )
        else:  # hacen lo mismo que por la derecha
            tex += '\\xrightarrow{' + latex( pasoDeEliminacion.subs(sust) ) + '}'
            tex += latex( factor((A & pasoDeEliminacion ).subs(sust)) )
        
    # transformaciones por la derecha
    for  _,pasoDeEliminacion in enumerate(pasos[1]):
        tex += '\\xrightarrow{' + latex( pasoDeEliminacion.subs(sust) ) + '}'
        tex += latex( factor((A & pasoDeEliminacion ).subs(sust)) )
                
    return tex


def rprElimFyC(data, pasos, TexPasosPrev=[], sust=[]):
    """Escribe en LaTeX los pasos efectivos y los sucesivos arreglos rectangulares"""
    if not data.es_arreglo_rectangular():
        raise ValueError('El sistema tiene que ser un arreglo rectangular.')
    if len(pasos[0])!=len(pasos[1]):
        raise ValueError('Esta representación requiere el mismo número de pasos por la izquierda y la derecha')
    
    A = data.fullcopy().subs(sust)
    tex = latex(data) if not TexPasosPrev else TexPasosPrev

    for  i,pasoDeEliminacionFilas in enumerate(pasos[0][::-1]):
        tex += '\\xrightarrow' \
                + '[' + latex( (pasoDeEliminacionFilas).subs(sust) ) + ']' \
                + '{' + latex( (pasos[1][i]).subs(sust)            ) + '}'
        tex += latex( factor(( pasoDeEliminacionFilas & A & pasos[1][i] )).subs(sust) )
                                                               
    return tex


def rprElimCF(data, pasos, TexPasosPrev=[], sust=[]):
    """Escribe en LaTeX los pasos efectivos y los sucesivos arreglos rectangulares"""
    if not data.es_arreglo_rectangular():
        raise ValueError('El sistema tiene que ser un arreglo rectangular')
    if len(pasos[0])!=len(pasos[1]):
        raise ValueError('Esta representación requiere el mismo número de pasos por la izquierda y la derecha')
    
    A = data.fullcopy().subs(sust)                                                               
    tex = latex(data) if not TexPasosPrev else TexPasosPrev

    for  i,pasoDeEliminacionFilas in enumerate(pasos[0][::-1]):
        tex += '\\xrightarrow{' + latex( (pasos[1][i]).subs(sust) ) + '}'
        tex += latex( factor((A & pasos[1][i]).subs(sust)) )
        tex += '\\xrightarrow[' + latex( (pasoDeEliminacionFilas).subs(sust) ) + ']{}' 
        tex += latex( factor((pasoDeEliminacionFilas & A).subs(sust)) )
                                                               
    return tex


def dispElim(self, pasos, TexPasosPrev=[]):
    display(Math(rprElim(self, pasos, TexPasosPrev)))

def dispElimFyC(self, pasos, TexPasosPrev=[]):
    display(Math(rprElimFyC(self, pasos, TexPasosPrev)))

def dispElimCF(self, pasos, TexPasosPrev=[]):
    display(Math(rprElimCF(self, pasos, TexPasosPrev)))


class Sistema:
    """Clase para listas ordenadas con reprentación latex
    
    Un Sistema es una lista ordenada de objetos. Los Sistemas se instancian
    con una lista, tupla u otro Sistema. 
    
    Parámetros:
        arg (list, tuple, Sistema): lista, tupla o Sistema de objetos.
    
    Atributos:
        lista (list): lista de objetos.
    
    Ejemplos:
    >>> # Crea un nuevo Sistema a partir de una lista, tupla o Sistema
    >>> Sistema( [ 10, 'hola', T({1,2}) ]  )           # con lista
    >>> Sistema( ( 10, 'hola', T({1,2}) )  )           # con tupla
    >>> Sistema( Sistema( [ 10, 'hola', T({1,2}) ] ) ) # con Sistema
    
    [10; 'hola'; T({1, 2});]
    
    """
    
    def __init__(self, arg):
        """Inicializa un Sistema con una lista, tupla o Sistema"""                        
        if es_ristra(arg):
            self.lista = list(arg)
        else:
            raise ValueError('El argumento debe ser una lista, tupla, o Sistema.')
    
        self.n            = len(self)
        self.corteSistema = set()
    
    
    def __getitem__(self, i):
        """ Devuelve el i-ésimo coeficiente del Sistema """
        return self.lista[i]
    
    def __setitem__(self, i, valor):
        """ Modifica el i-ésimo coeficiente del Sistema """
        self.lista[i] = valor
            
    
    def __len__(self):
        """Número de elementos del Sistema """
        return len(self.lista)
    
    
    def copy(self):
        """ Genera un Sistema copiando la lista de otro """
        return type(self)(self.lista)
    
    
    def __eq__(self, other):
        """Indica si es cierto que dos Sistemas son iguales"""
        return self.lista == other.lista
    
    def __ne__(self, other):
        """Indica si es cierto que dos Sistemas son distintos"""
        return self.lista != other.lista
    
    
    def reverse(self):
        """Da la vuelta al orden de la lista del sistema"""
        self.corteSistema =  {len(self)-i for i in self.corteSistema}
        self.lista.reverse()
        
    def __reversed__(self):
        """Reversed(S) devuelve una copia de S con la lista en orden inverso"""
        copia = self.fullcopy()
        copia.reverse()
        return copia
        
    
    def concatena(self, other, marcasVisuales = False):
        """Concatena dos Sistemas"""    
        def nuevoConjuntoMarcas(Sistema_A, Sistema_B):
            return Sistema_A.corteSistema.union(
                {len(Sistema_A)},
                {len(Sistema_A)+indice for indice in Sistema_B.corteSistema} )
        
        if not isinstance(other, Sistema):
            raise ValueError('Un Sistema solo se puede concatenar a otro Sistema')
    
        if self:
            sistemaAmpliado = self.fullcopy()
        else:
            return other.fullcopy()
            
        sistemaAmpliado.lista = self.lista + other.lista
        sistemaAmpliado.n     = len(self)  + len(other)
            
        if marcasVisuales: 
            sistemaAmpliado.corteSistema = nuevoConjuntoMarcas(self, other)
    
        return sistemaAmpliado if self.es_arreglo_rectangular() else Sistema(sistemaAmpliado)
    
    
    def junta(self, lista, marcas = False):
        """Junta una lista o tupla de Sistemas en uno solo concatenando las
        correspondientes listas de los distintos Sistemas
    
        """
        reune = lambda lista,marcas: lista[0] if len(lista)==1 else lista[0].concatena(reune(lista[1:],marcas), marcas)    
        return reune([self] + [sistema for sistema in lista], marcas)
        
    
    def amplia(self, args, marcas = False):
        """Añade más elementos al final de la lista de un Sistema"""
        A = self.fullcopy()
        return A.concatena(Sistema(CreaLista(args)), marcas)
    
    
    def subs(self, reglasDeSustitucion=[]):
        """ Sustitución de variables simbólicas """
        reglas = CreaLista(reglasDeSustitucion)
        self.lista = [ sympy.S(elemento).subs(CreaLista(reglas)) for elemento in self.lista ]
        return self
    
    
    def simplify(self):
        """ Simplificación de expresiones simbólicas """
        self.lista = [ simplify(elemento) for elemento in self.lista ]
                                                                   
    def factor(self):
        """ Factorización de expresiones simbólicas """
        self.lista = [ factor(elemento) for elemento in self.lista ]
    
    def expand(self):
        """ Factorización de expresiones simbólicas """
        self.lista = [ expand(elemento) for elemento in self.lista ]
    
    
    
    def fullcopy(self):
        """ Copia la lista de otro Sistema y sus atributos"""
        new_instance = self.copy()
        new_instance.__dict__.update(self.__dict__)
        return new_instance
    
    
    def sis(self):
        """Devuelve el Sistema en su forma genérica"""
        return Sistema(self.lista)
    
    
    def es_nulo(self, sust=[]):
        """Indica si es cierto que el Sistema es nulo"""
        return self.subs(sust) == self*0
    
    def no_es_nulo(self, sust=[]):
        """Indica si es cierto que el Sistema no es nulo"""
        return self.subs(sust) != self*0
    
    
    def es_arreglo_rectangular(self):
        """Indica si el Sistema tiene estructura de arreglo rectangular"""
    
        def solo_contiene_sistemas(sis):
            return all([isinstance(elemento, Sistema) for elemento in sis])
    
        def elementos_con_la_misma_logitud(sis):
            primerElemento = sis|1
            return all([len(primerElemento)==len(elemento) for elemento in sis])
    
        if solo_contiene_sistemas(self) and elementos_con_la_misma_logitud(self):
            return True
        else:
            return False
    
    def no_es_arreglo_rectangular(self):
        """Indica si el Sistema no tiene estructura de arreglo rectangular"""
        return not self.es_arreglo_rectangular()
    
    
    def es_de_composicion_uniforme(self):
       """Indica si es cierto que todos los elementos son del mismo tipo"""
       if all([es_numero(c) for c in self]):
          return True
       else:
          return all(type(elemento)==type(self|1) for elemento in self)
    
    
    def es_de_composicion_y_longitud_uniforme(self):
       """Indica si es cierto que todos los elementos son del mismo tipo y
       longitud
    
       """
       if self.es_de_composicion_uniforme() and es_numero(self|1):
          return True
       elif self.es_de_composicion_uniforme() and not es_numero(self|1):
          return all(len(elemento)==len(self|1) for elemento in self)
       else:
          return False
       
    
    def primer_no_nulo(self, reglasDeSustitucion=[]):
        """Devuelve una lista con la posición del primer no nulo o vacía si
        todos los elementos son nulos
    
        """
        sistema = self.subs(reglasDeSustitucion)
        return next( ([indice] for indice, elemento in enumerate(sistema, 1) if CreaSistema(elemento).no_es_nulo()), [])
    
    def ultimo_no_nulo(self, reglasDeSustitucion=[]):
        """Devuelve una lista con la posición del primer no nulo o vacía si
        todos los elementos son nulos
    
        """
        sistema = reversed(self.copy()).subs(reglasDeSustitucion)
        return next( ([len(self)-indice] for indice,elemento in enumerate(sistema) if CreaSistema(elemento).no_es_nulo()), [])
    
    elementoPivote     = lambda self:  self.extractor(self.primer_no_nulo()) if self.primer_no_nulo() else False
    
    elementoAntiPivote = lambda self:  self.extractor(self.ultimo_no_nulo()) if self.ultimo_no_nulo() else False
    
    
    def extractor(self, listaDeIndices = []):
        """Selección consecutiva por la derecha del sistema A empleando la
        lista de enteros de c. Ej.: si c = [5,1,2] devuelve A|5|1|2
    
        """
        objeto = self
        for indice in listaDeIndices:
            objeto = objeto|indice
        return objeto if listaDeIndices else []
    
       
    def reshape(self, orden=[]):
        "Reordena los elementos de un Sistema para generar un BlockM"
        if not orden or isinstance(orden, int):
            return self
        elif orden[0]*orden[1] == self.n:
            return ~BlockM(list(zip(*(iter(self.lista),) * orden[0])))
        else:
            raise ValueError('El orden indicado es incompatible con el número de elementos del Sistema')
            return None
    
    
    def span(self, sust=[], Rn=[]):
        return SubEspacio(self.sis(), sust, Rn)
    
    
    def espacio_nulo(self, sust=[], Rn=[]):
        if self: Rn = self.n
        K     = self.elim(0, False, sust)
        E     = I(self.n) & T(K.pasos[1])
        lista = [v for j,v in enumerate(E,1) if (K|j).es_nulo()]
        return SubEspacio(Sistema(lista)) if lista else SubEspacio(Sistema([]), Rn=Rn)
    
    
    def sel(self, v, sust=[]):
        """Devuelve la lista o EAfin con las soluciones x de sistema*x=v
    
        """
        A           = self.amplia(-v)
        operaciones = A.elim(1,False,sust).pasos[1]
        testigo     = 0| (I(A.n) & T(operaciones)) |0
        Normaliza   = T([]) if testigo==1 else T([( fracc(1,testigo), A.n )])
        pasos       = operaciones+[Normaliza] if Normaliza else operaciones
        K           = A & T(pasos)
        
        if (K|0).no_es_nulo():
            return Sistema([])
        else:
            solP = factor(I(self.n).amplia(V0(self.n)) & T(pasos))|0
            if self.espacio_nulo().sgen.es_nulo():
                return Sistema([solP])
            else:
                return EAfin(self.espacio_nulo().sgen, solP, 1)
    
    
    def __or__(self,j):
        """Extrae el j-ésimo componente del Sistema; o crea un Sistema con la
        tupla de elementos indicados (los índices comienzan por el número 1)
        
        Parámetros:
            j (int, list, tuple, slice): Índice (o lista de índices) del 
                  elementos (o elementos) a seleccionar
        
        Resultado:
                  ?: Si j es int, devuelve el elemento j-ésimo del Sistema.
            Sistema: Si j es list, tuple o slice devuelve el Sistema formado por
                  los elementos indicados en la lista, tupla o slice de índices.
        
        Ejemplos:
        >>> # Extrae el j-ésimo elemento del Sistema 
        >>> Sistema([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | 2
        
        Vector([0, 2])
        
        >>> # Sistema formado por los elementos indicados en la lista (o tupla)
        >>> Sistema([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | [2,1]
        >>> Sistema([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | (2,1)
        
        [Vector([0, 2]); Vector([1, 0])]
        
        >>> # Sistema formado por los elementos indicados en el slice
        >>> Sistema([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | slice(1,3,2)
        
        [Vector([1, 0]), Vector([3, 0])]
        
        """
        if isinstance(j, int):
            return self[j-1]
            
        elif isinstance(j, (list,tuple) ):
            return type(self) ([ self|indice for indice in j ])
        
        elif isinstance(j, slice):
            start = None if j.start is None else j.start-1 
            stop  = None if j.stop  is None else (j.stop if j.stop>0 else j.stop-1)
            step  = j.step  or 1
            return type(self) (self[slice(start,stop,step)])
    
    
    def __ror__(self,i):
        """Hace exactamente lo mismo que el método __or__ por la derecha."""
        return self | i
    
    
    def __add__(self, other):
        
        if not type(self)==type(other) or not len(self)==len(other):
            raise ValueError ('Solo se suman Sistemas del mismo tipo y misma longitud')
        suma = self.fullcopy()
        suma.lista = [ (self|i) + (other|i) for i in range(1,len(self)+1) ]
        suma.corteSistema.update(other.corteSistema)
        return factor(suma)
                
    def __sub__(self, other):
        """Devuelve el Sistema resultante de restar dos Sistemas
        
        Parámetros: 
            other (Sistema): Otro sistema del mismo tipo y misma longitud
        
        Ejemplos:
        >>> Sistema([10, 20, 30]) - Sistema([1, 1, -1])
        
        Sistema([9, 19, 31])
        >>> Vector([10, 20, 30]) - Vector([1, 1, -1])
        
        Vector([9, 19, 31])
        >>> Matrix([[1,5],[5,1]]) - Matrix([[1,0],[0,1]]) 
        
        Matrix([Vector([0, 5]); Vector([5, 0])]) 
        """
        if not type(self)==type(other) or not len(self)==len(other):
            raise ValueError ('Solo se restan Sistemas del mismo tipo y misma longitud')
        diferencia = self.fullcopy()
        diferencia.lista = [ (self|i) - (other|i) for i in range(1,len(self)+1) ]
        diferencia.corteSistema.update(other.corteSistema)
        return factor(diferencia)
                
    
    def __rmul__(self, x):
        """Multiplica un Sistema por un número a su izquierda
        
        Parámetros:
            x (int, float o sympy.Basic): Escalar por el que se multiplica
        Resultado:
            Sistema resultante de multiplicar cada componente por x
        Ejemplo:
        >>> 3 * Sistema([10, 20, 30]) 
        
        Sistema([30, 60, 90]) 
        """
        if es_numero(x):
            multiplo = self.fullcopy()
            multiplo.lista = [ x*(self|i) for i in range(1,len(self)+1) ]
            return factor(multiplo)
    
    
    def __neg__(self):
        """Devuelve el opuesto de un Sistema"""
        return -1*self
    
    
    def __mul__(self,x):
        """Multiplica un Sistema por un número, Vector o una Matrix a su derecha
        
        Parámetros:
            x (int, float o sympy.Basic): Escalar por el que se multiplica
              (Vector): con tantos componentes como el Sistema
              (Matrix): con tantas filas como componentes tiene el Sistema
        
        Resultado:
            Sistema del mismo tipo: Si x es int, float o sympy.Basic, devuelve 
               el Sistema que resulta de multiplicar cada componente por x
            Objeto del mismo tipo de los componentes del Sistema: Si x es Vector,
               devuelve una combinación lineal de los componentes del Sistema, 
               donde los componentes de x son los coeficientes de la combinación.
            Sistema del mismo tipo: Si x es Matrix, devuelve un Sistema cuyas 
               componentes son combinación lineal de las componentes originales.
               
        Ejemplos:
        >>> # Producto por un número
        >>> Vector([10, 20, 30]) * 3
        
        Vector([30, 60, 90])
        >>> Matrix([[1,2],[3,4]]) * 10
        
        Matrix([[10,20],[30,40]])
        >>> # Producto por un Vector
        >>> Vector([10, 20, 30]) * Vector([1, 1, 1])
        
        60
        >>> Matrix([Vector([1, 3]), Vector([2, 4])]) * Vector([1, 1])
        
        Vector([3, 7])
        >>> # Producto por una Matrix
        >>> Vector([1,1,1])*Matrix( ( [1,1,1], [2,4,8], [3,-1,0] ) )
        
        Vector([6, 4, 9])
        >>> Matrix([Vector([1, 3]), Vector([2, 4])]) * Matrix([Vector([1,1])]))
        
        Matrix([Vector([3, 7])])
        
        """
        if es_numero(x):
            return x*self
    
        elif isinstance(x, Vector):
            if len(self) != x.n:
                raise ValueError('Sistema y Vector incompatibles')
            if self.es_arreglo_rectangular():
                if not all([f.es_de_composicion_y_longitud_uniforme() for f in ~BlockM([BlockV([i]) for i in self])]):
                    raise ValueError('El sistema de la derecha debe tener elementos de composicion y longitud uniforme')
            elif not self.es_de_composicion_y_longitud_uniforme():
                raise ValueError('El sistema de la derecha debe tener elementos de composicion y longitud uniforme')
                
            return factor(sum([(self|j)*(x|j) for j in range(1,len(self)+1)], 0*self|1))
        
        elif isinstance(x, Matrix):
            if len(self) != x.m:
                raise ValueError('Sistema y Matrix incompatibles')
            if isinstance(self, BlockV):
                return factor(BlockV( [ self*(x|j) for j in range(1,(x.n)+1)], rpr='fila' ))
            elif isinstance(self, BlockM):
                return factor(BlockM ( [ self*(x|j) for j in range(1,(x.n)+1)] ))
            else:
                return factor(type(self) ( [ self*(x|j) for j in range(1,(x.n)+1)] ))
    
    
    def __and__(self,operaciones):
        """Transforma los elementos de un Sistema 
        
            T(abreviaturas): transformaciones a aplicar sobre un Sistema S
        Ejemplos:
        >>>  S & T({1,3})                # Intercambia los elementos 1º y 3º
        >>>  S & T((5,1))                # Multiplica por 5 el primer elemento
        >>>  S & T((5,2,1))              # Suma 5 veces el 2º elem al 1º
        >>>  S & T([{1,3},(5,1),(5,2,1)])# Aplica la secuencia de transformac.
                     # sobre los elementos de S y en el orden de la lista
        """
        def transformacionDelSistema(abrv):
            if isinstance(abrv,set):
                self.lista = [ (self|max(abrv)) if k==min(abrv) else \
                               (self|min(abrv)) if k==max(abrv) else \
                               (self|k)                 for k in range(1,len(self)+1)].copy()
                
            elif isinstance(abrv,tuple) and (len(abrv) == 2):
                self.lista = [ (abrv[0])*(self|k) if k==abrv[1] else (self|k) \
                                                        for k in range(1,len(self)+1)].copy()
    
            elif isinstance(abrv,tuple) and (len(abrv) == 3):
                colPivote = abrv[1]-1
                self.lista = [ (abrv[0])*(self.lista[colPivote]) + (self|k) if k==abrv[2] else (self|k)
                                                        for k in range(1,len(self)+1)].copy()
    
        for abrv in operaciones.abreviaturas:
            transformacionDelSistema(abrv)
    
        return factor(self)
            
            
    def __rand__(self, operaciones):
        """Hace exactamente lo mismo que el método __and__ por la derecha."""
        return self & operaciones
        
    
    
    def elim(self, variante=0, rep=False, sust=[], repsust=False):
        """Versión pre-escalonada de un sistema por eliminacion Derecha-Izquierda"""
        
        def texYpasos(data, pasos, rep=0, sust=[], repsust=0):
            pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
            TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
            if repsust:
                tex = rprElim(data, pasos, TexPasosPrev, sust)
            else:
                tex = rprElim(data, pasos, TexPasosPrev)
            pasos[0] = pasos[0] + pasosPrevios[0] 
            pasos[1] = pasosPrevios[1] + pasos[1]
            
            if rep:
                display(Math(tex))
            
            return tex, pasos
        
        def metodos_auxiliares_de_la(variante):
            """Define los métodos auxilares y el módo de actuació sobre el sistema
            en función de la variante de elimiación elegida.
        
            'variante' es la suma de los siguientes números:
                
               +1 reduccion rápida (solo transformaciones tipo I)
               +2 doble reducción
               +4 por filas
              +10 normalización de los pivotes
              +20 escalonamiento
             +100 de atrás hacia delante
            
            Por defecto arg = 0 (reducción simple hacia delante, por
            columnas y evitando fraciones)
        
            """
            if 100 in analisis_opcion_elegida(variante): # reducción hacia delante
                componentesAmodificar     = lambda    sistema:  filter(lambda x:  x < indiceXP, range(1,len(sistema)+1))
                recorrido                 = lambda    sistema:  reversed(list(enumerate(CreaSistema(sistema),1)))
                XPivote                   = lambda componente:  elementoAntiPivote(componente)
                posicionXPivote           = lambda componente:  ultimo_no_nulo(componente)
                        
            else:                                    # reducción hacia atrás
                componentesAmodificar     = lambda    sistema:  filter(lambda x:  x > indiceXP, range(1,len(sistema)+1))
                recorrido                 = lambda    sistema:  enumerate(CreaSistema(sistema),1)
                XPivote                   = lambda componente:  elementoPivote(componente)
                posicionXPivote           = lambda componente:  primer_no_nulo(componente)
                
            if 4 in analisis_opcion_elegida(variante):   # reducción de los componentes en arreglos rectangulares
                if (not self.es_arreglo_rectangular()) or (not all([item.es_de_composicion_uniforme() for item in self])):
                    raise ValueError('El sistema debe ser un arreglo rectangular con componentes de composición uniforme')
                sistema = ~self.fullcopy().subs(sust);
            else:
                sistema = self.fullcopy().subs(sust);
        
            if 2 in analisis_opcion_elegida(variante):   # doble reducción (reducción posiciones anteriores y posteriores al pivote)
                componentesAmodificar = lambda    sistema:  filter(lambda x: x != indiceXP, range(1,len(sistema)+1))
        
            return sistema, recorrido, XPivote, posicionXPivote, componentesAmodificar
        
            
        def Reduccion(sistema):
            if 1 in analisis_opcion_elegida(variante):   # reducción rápida (solo trasformaciones tipo I)
                operaciones = [ (-fracc(ValorAEliminar(indiceVAE), pivote), indiceXP, indiceVAE)  \
                                                            for indiceVAE in componentesAmodificar(sistema)]
            else:                                        # reducción lenta (evitando fracciones)
                operaciones = [[( denom(ValorAEliminar(indiceVAE), pivote),           indiceVAE), \
                                (-numer(ValorAEliminar(indiceVAE), pivote), indiceXP, indiceVAE)] \
                                                            for indiceVAE in componentesAmodificar(sistema)]
            return filtradopasos(T(operaciones))
        
        def Normalizacion(sistema):
            return filtradopasos(T([ (fracc(1, XPivote(sistema|indiceXP)), indiceXP)
                                     for indiceXP,_ in recorrido(sistema) if XPivote(sistema|indiceXP)]))
        
        def Escalonamiento(sistema):
            M = sistema.copy()
            if 100 in analisis_opcion_elegida(variante): # con reducción hacia atrás
                destino       = lambda     : (M.n)-r+1
                resto         = lambda    r: slice(None, max(M.n-r,1))
                columnaAMover = lambda i, r: posicionXPivote(i|M|resto(r))[0]   if posicionXPivote(i|M|resto(r)) and i==posicionXPivote(M|posicionXPivote(i|M|resto(r))[0]  )[0] else 0
            else:                                        # con reducción hacia delante
                destino       = lambda     : r
                resto         = lambda    r: slice(r+1, None)
                columnaAMover = lambda i, r: posicionXPivote(i|M|resto(r))[0]+r if posicionXPivote(i|M|resto(r)) and i==posicionXPivote(M|posicionXPivote(i|M|resto(r))[0]+r)[0] else 0
        
            r = 0
            intercambios = []
            for i,_ in recorrido(M|1):
                indiceColumnaPivote = columnaAMover(i,r)
                if indiceColumnaPivote:
                    r += 1
                    intercambio  = T( {destino(), indiceColumnaPivote} )
                    M & intercambio
                    intercambios.append(intercambio)
                    
            return filtradopasos(T(intercambios))
        
        
        def transformacionYPasos(sistema, operacion, pasosPrevios):
            pasoDado = operacion(sistema)
            if 4 in analisis_opcion_elegida(variante):    # reducción de los componentes en arreglos rectangulares
                pasosAcumulados = [~pasoDado] + pasosPrevios if pasoDado else pasosPrevios
            else:
                pasosAcumulados = pasosPrevios  + [pasoDado] if pasoDado else pasosPrevios
            sistema & T(pasoDado)
            return factor(sistema.subs(sust)), pasosAcumulados
        
        
        def sistemaFinalYPasosDchaIzda(sistema,transformaciones):
            if 4 in analisis_opcion_elegida(variante):    # reducción de los componentes en arreglos rectangulares
                TransformacionesPorLaIzquierda = filtradopasos(transformaciones)
                TransformacionesPorLaDerecha   = []
                if self.es_arreglo_rectangular():
                    sistema = ~sistema
            else: 
                TransformacionesPorLaDerecha   = filtradopasos(transformaciones)
                TransformacionesPorLaIzquierda = []
        
            SistemaFinal =  sistema.subs(sust)
            pasos        = [TransformacionesPorLaIzquierda, TransformacionesPorLaDerecha]
            SistemaFinal.tex, SistemaFinal.pasos = texYpasos(self, pasos, rep, sust, repsust)
            SistemaFinal.TrF = T(SistemaFinal.pasos[0])
            SistemaFinal.TrC = T(SistemaFinal.pasos[1])
            return factor(SistemaFinal)
        
        
        def analisis_opcion_elegida(tipo):
            'Análisis de las opciones de eliminación elegidas'
            lista = [100,20,10,4,2,1]
            opcion = set()
            for t in lista:
                if (tipo - (tipo % t)) in lista:
                    opcion.add(tipo - (tipo % t))
                    tipo = tipo % t
            return opcion
            
        
        if not self:
            return sistemaFinalYPasosDchaIzda(Sistema([]), [T([])] )
        
        if not self.es_de_composicion_y_longitud_uniforme():
            raise ValueError('Los elementos del sistema deben ser del mismo tipo y longitud')
        
        ValorAEliminar = lambda indiceVAE: sistema.extractor([indiceVAE]+posicionXPivote(sistema|indiceXP))
        sistema, recorrido, XPivote, posicionXPivote, componentesAmodificar = metodos_auxiliares_de_la(variante)
        
        pasosAcumulados = []
        for indiceXP,_ in recorrido(sistema):
            pivote = XPivote(sistema|indiceXP)        
            if pivote:                               # reducción
                sistema, pasosAcumulados = transformacionYPasos(sistema,  Reduccion,  pasosAcumulados)
                
        if 10 in analisis_opcion_elegida(variante):  # normalización de pivotes
            sistema, pasosAcumulados = transformacionYPasos(sistema,  Normalizacion,  pasosAcumulados)
    
        if 20 in analisis_opcion_elegida(variante):  # escalonamiento
            sistema, pasosAcumulados = transformacionYPasos(sistema, Escalonamiento,  pasosAcumulados)
    
        return sistemaFinalYPasosDchaIzda(sistema, pasosAcumulados)
            
    
    def K(self,rep=0, sust=[], repsust=0):
        """Una forma pre-escalonada por columnas (K) de una Matrix"""
        return self.elim(0, rep, sust, repsust)
        
    def L(self,rep=0, sust=[], repsust=0): 
        """Una forma escalonada por columnas (L) de una Matrix"""
        return self.elim(20, rep, sust, repsust)
        
    def R(self,rep=0, sust=[], repsust=0):
        """Forma escalonada reducida por columnas (R) de una Matrix"""
        return self.elim(32, rep, sust, repsust)
    
    def U(self,rep=0, sust=[], repsust=0): 
        """Una forma escalonada por filas (U) de una Matrix"""
        return self.elim(24, rep, sust, repsust)
    
    def UR(self,rep=0, sust=[], repsust=0): 
        """Una forma escalonada reducida por filas (U) de una Matrix"""
        return self.elim(36, rep, sust, repsust)
    
    
    def __str__(self):
        """ Muestra un Sistema en su representación python """
        pc = ';' if len(self.lista) else ''
        ln = [len(n) for n in particion(self.corteSistema,self.n)]
        return '[' + \
                 ';|'.join(['; '.join([str(c) for c in s]) \
                           for s in [ self|i for i in particion(self.corteSistema, self.n)] ]) + pc + ']'
    
    
    def __repr__(self):
        """ Muestra un Sistema en su representación python """
        pc = ';' if len(self.lista) else ''
        return 'Sistema([' + '; '.join( repr (e) for e in self ) + pc + '])'
    
    
    def latex(self):
        """ Construye el comando LaTeX para representar un Sistema """
        pc = ';' if len(self) else r'\ '
        ln = [len(i) for i in particion(self.corteSistema, len(self))]                                                           
        return \
            r'\left[ \begin{array}{' + '|'.join([n*'c' for n in ln])  + '}' + \
            r';& '.join([latex(e) for e in self]) + pc + \
            r'\end{array} \right]'
    
    def _repr_html_(self):
        """ Construye la representación para el entorno jupyter notebook """
        return html(self.latex())
    
    def _repr_latex_(self):
        """ Representación para el entorno jupyter en Emacs """
        return '$'+self.latex()+'$'
    
    def _repr_png_(self):
        """ Representación png para el entorno jupyter en Emacs """
        try:
            expr = '$'+self.latex()+'$'
            workdir = tempfile.mkdtemp()
            with open(join(workdir, 'borrame.png'), 'wb') as outputfile:
                sympy.preview(expr, viewer='BytesIO', outputbuffer=outputfile)
            return open(join(workdir, 'borrame.png'),'rb').read()
        except:
            return '$'+self.latex()+'$'
                                                                   
    
    def ccol(self, conjuntoIndices={}):
        """Modifica el atributo corteSistema para insertar lineas entre
        determinados elementos del sistema
    
        """
        self.corteSistema = set(conjuntoIndices) if conjuntoIndices else {0}
        return self
    
    
RistraTypes = (tuple, list, Sistema)
es_ristra  = lambda x: isinstance(x, RistraTypes) 

def es_ristra_de_numeros(arg):
    return all( [es_numero(elemento) for elemento in arg] ) if es_ristra(arg) else None

class BlockV(Sistema):
    """BlockV es un Sistema que se puede representar verticalmente.
    
    Se puede instanciar con una lista, tupla o otro Sistema. Si al
    instanciar un BlockV la lista, tupla o sistema solo contiene números
    el objeto obtenido es un Vector (subclase de BlockV).
    
    El atributo 'rpr' indica si la representación latex debe mostrar el
    sistema en disposición vertical (por defecto) u horizontal.
    
    Parámetros:
        sis   (list, tuple, Sistema): Lista, tupla o Sistema de objetos.
        rpr   (str): Para su representación latex (en vertical por defecto).
                      Si rpr='fila' se representa en forma horizontal. 
    
    Atributos de la subclase:
        rpr   (str): modo de representación en Jupyter.
    
    Atributos heredados de la clase Sistema:
        lista              (list): list con los elementos.
        n                  (int) : número de elementos de la lista.
        corteSistema (set) : Conjunto de índices donde pintar
                                    separaciones visuales
    
    Ejemplos:
    >>> # Instanciación a partir de una lista, tupla o Sistema de números
    >>> BlockV( [1,'abc',(2,)] )            # con una lista
    >>> BlockV( (1,'abc',(2,)) )            # con una tupla
    >>> BlockV( Sistema( [1,'abc',(2,)] ) ) # con un Sistema
    >>> BlockV( BlockV ( [1,'abc',(2,)] ) ) # a partir de otro BlockV
    
    BlockV( [1,'abc',(2,)] )
    
    >>> BlockV( [1,2,3)] )                  # con una lista de números
    
    Vector( [1,2,3] )
    """
    
    def __init__(self, arg, rpr='columna'):
        """Inicializa un BlockV con una lista, tupla o Sistema"""
        super().__init__(arg)
        self.rpr  =  rpr
        self.n  = len(self)
        
        if all( [es_numero(e) for e in arg] ): self.__class__ = Vector
    
    
    def __repr__(self):
        """ Muestra el BlockV en su representación Python """
        return 'BlockV(' + repr(self.lista) + ')'
                               
    def __str__(self):
        """ Muestra el BlockV en su representación Python """
        pc = ',' if len(self.lista) else ''
        ln = [len(n) for n in particion(self.corteSistema,self.n)]
        return '(' + \
            ',|'.join([', '.join([str(c) for c in s]) \
                       for s in [ self|i for i in particion(self.corteSistema, self.n)]]) + \
            pc + ')'
    
    def latex(self):
        """ Construye el comando LaTeX para representar un BlockV """
        if bool(self.corteSistema):
            pc = ',' if len(self) else r'\ '
            ln = [len(n) for n in particion(self.corteSistema,self.n)]
            if self.rpr == 'fila' or self.n==1:    
                return \
                    r'\left( \begin{array}{' + '|'.join([n*'c' for n in ln])  + '}' + \
                    r',& '.join([latex(e) for e in self]) + pc + \
                    r'\\ \end{array} \right)'
            else:
                return \
                    r'\left( \begin{array}{c}' + \
                    r'\\ \hline '.join([r'\\'.join([latex(c) for c in e]) \
                        for e in [ self|i for i in particion(self.corteSistema, self.n)]]) + \
                    r'\\ \end{array} \right)'
        else:
            if self.rpr == 'fila' or self.n==1:
                return r'\begin{pmatrix}' + \
                    ',& '.join([latex(e) for e in self]) + \
                    r',\end{pmatrix}'
            else:
                return r'\begin{pmatrix}' + \
                    r'\\ '.join([latex(e) for e in self]) + \
                    r'\end{pmatrix}'
    
class Vector(BlockV):
    """Clase para los Sistemas de números.
    
    Sólo se puede instanciar con una lista, tupla o Sistema de objetos
    int, float o sympy.Basic. Si se instancia con un Vector se crea una
    copia del mismo.
    
    El atributo 'rpr' indica si, en la representación latex, el vector
    debe ser escrito como columna (por defecto) o como fila.
    
    Parámetros:
        sis (list, tuple, Sistema): Lista, tupla o Sistema de objetos
            de tipo int, float o sympy.Basic.
        rpr (str): Para su representación latex (en 'columna' por defecto).
            Si rpr='fila' el vector se representa en forma de fila. 
    
    Atributos heredados de la subclase BlockV::
        rpr   (str)    : modo de representación en Jupyter.
    
    Atributos heredados de la clase Sistema:
        lista              (list): list con los elementos.
        n                  (int) : número de elementos de la lista.
        corteSistema (set) : Conjunto de índices donde pintar
                                    separaciones visuales
    
    Ejemplos:
    >>> # Instanciación a partir de una lista, tupla o Sistema de números
    >>> Vector( [1,2,3] )           # con lista
    >>> Vector( (1,2,3) )           # con tupla
    >>> Vector( Sistema( [1,2,3] ) )# con Sistema
    >>> Vector( Vector ( [1,2,3] ) )# a partir de otro Vector
    
    Vector([1,2,3])
    """
    
    def __init__(self, arg, rpr='columna'):
        """Inicializa Vector con una lista, tupla o Sistema de números"""                       
        if not es_ristra_de_numeros(arg):
            raise ValueError('no todos los elementos son números o parámetros!')
    
        super().__init__(arg, rpr)
    
    
    def norma(self):
        """Devuelve la norma de un vector"""
        return sympy.sqrt(self*self)
                                                                   
    def normalizado(self):
        """Devuelve un múltiplo de norma uno si el vector no es nulo"""
        if self.es_nulo(): raise ValueError('Un vector nulo no se puede normalizar')
        return self * fracc(1,self.norma())
    
    
    def diag(self):
        """Crea una Matrix diagonal cuya diagonal es self"""
        return Matrix([a*(I(self.n)|j) for j,a in enumerate(self, 1)])
    
    
    def __repr__(self):
        """ Muestra el vector en su representación Python repr """
        return 'Vector(' + repr(self.lista) + ')'
    

class V0(Vector):
    """Clase para los Vectores nulos"""
    def __init__(self, n, rpr = 'columna'):
        """Inicializa un vector nulo de n componentes

        V0 se inicializa con

        1) un entero que indica el  número de componentes nulas
    
        2) la cadena de texto rpr. Si rpr = 'fila' la representación
        es horizontal (en otro caso es vertical)

        """
        super().__init__([0]*n, rpr)
        self.__class__ = Vector

class V1(Vector):
    """Clase para los Vectores constantes 1"""
    def __init__(self, n, rpr = 'columna'):
        """Inicializa un vector uno de n componentes

        V1 se inicializa con

        1) un entero que indica el  número de componentes nulas
    
        2) la cadena de texto rpr. Si rpr = 'fila' la representación
        es horizontal (en otro caso es vertical)

        """
        super().__init__([1]*n, rpr)
        self.__class__ = Vector

class BlockM(Sistema):
    """Clase para arreglos rectangulares de objetos.
    
    Sistema formado por BlockVs con el mismo número de componentes. Se
    instancia con: 1) una lista, tupla o Sistema de BlockVs (serán sus
    columnas); 2) una lista, tupla o Sistema de listas, tuplas o Sistemas
    con la misma longitud (serán sus filas); 3) otro BlockM (se obtendrá
    una copia).
    
    Parámetros:
        arg (list, tuple, Sistema): Lista, tupla o Sistema de BlockVs
            (con identica longitud); o de listas, tuplas o Sistemas (con
            identica longitud).
    
    Atributos:
        m              (int) : número de filas
        corteElementos (set) : Conjunto de índices donde pintar
                                separaciones visuales entre filas
    
    Atributos heredados de la clase Sistema:
        lista              (list): list con los elementos.
        n                  (int) : número de elementos de la lista.
        corteSistema (set) : Conjunto de índices donde pintar
                                    separaciones visuales
    
    Ejemplos:
    >>> # Crear un BlockM a partir de una lista de Vectores o BlockVs:
    >>> a = BlockV( ['Hola',2,3] ); b = Vector( [1,2,5] ); c = Vector( [1,2,7] )
    >>> BlockM( [a,b,c] )
    
    BlockM([BlockV(['Hola', 2, 3]), Vector([1, 2, 5]), Vector([1, 2, 7])])
    >>> # Crear una BlockM a partir de una lista de listas, tuplas o Sistemas
    >>> A = BlockM([('Hola',1,1),[2,2,2],Vector([3,5,7])])
    
    BlockM([BlockV(['Hola', 2, 3]), Vector([1, 2, 5]), Vector([1, 2, 7])])
    
    """
    
    def __init__(self, arg):
        """Inicializa un BlockM con una
    
        1) lista, tupla o Sistema de BlockVs con el mismo número de
        elementos,
        
        2) tupla, lista o Sistema de tuplas, listas o Sistemas con el
        mismo número de elementos,
        
        3) con otra BlockM.
    
        """
        super().__init__(arg)
        
        lista = Sistema(arg).lista
        
        if all([(isinstance(elemento, BlockV) and len(elemento)==len(lista[0])) for elemento in lista]):
            self.lista   = lista.copy()
    
        elif all([(es_ristra(elemento) and len(elemento)==len(lista[0])) for elemento in lista]):
    
            self.lista = BlockM([ BlockV([ elemento[i] for elemento in lista ]) for i in range(len(lista[0])) ]).lista
    
        elif isinstance(arg, BlockM):
            self.lista   = arg.lista.copy()
    
        else: 
            raise ValueError("""El argumento debe ser una lista de BlockVs
            o una lista, tupla o Sistema de listas, tuplas o sistemas con
            el mismo número de elementos!""")
        
        self.n  = len(self)
        try: 
            self.m  = (self|1).n
        except:
            self.m  = 0
                
        self.corteElementos = set()
       
        for v in self.lista:
            v.rpr='columna'
            
        if all( [isinstance(e,Vector) for e in self] ): self.__class__ = Matrix
        
    
    def __invert__(self):
        """
        Devuelve la traspuesta de un BlockM.
        
        Ejemplo:
        >>> ~BlockM([ [1,2,3], [2,4,6] ])
        
        Matrix([ Vector([1, 2, 3]), Vector([2, 4, 6]) ])
        """
        M = BlockM([ Sistema(columna) for columna in self ])
        M.corteElementos, M.corteSistema = self.corteSistema, self.corteElementos
        return M
    
    
    def columnas_homogeneas(self):
        """Indica si las columnas contienen objetos del mismo tipo y longitud"""
        return self.es_de_composicion_y_longitud_uniforme()
    
    def filas_homogeneas(self):
        """Indica si las filas contienen objetos del mismo tipo y longitud"""
        return (~self).es_de_composicion_y_longitud_uniforme()
    
    
    def apila(self, lista, marcasVisuales = False):
        """Apila una lista o tupla de BlockMs con el mismo número de elementos
        (columnas) en un BlockM concatenando los respectivos elementos
    
        """
        apila_dos = lambda x, other, marcasVisuales=False: ~((~x).concatena(~other, marcasVisuales))
        apila = lambda x: x[0] if len(x)==1 else apila_dos( apila(x[0:-1]), x[-1], marcasVisuales)
        return apila([self] + [s for s in CreaLista(lista)])
    
    
    def __ror__(self,i):
        """Extrae la j-ésima fila de un BlockM en forma de BlockV; o crea un
        BlockM cuyas filas corresponden a las filas indicadas en una tupla o
        lista de índices (los índices comienzan por el número 1)
        
        Parámetros:
            j (int, list, tuple, slice): Índice (o lista de índices) del 
                  elementos (o elementos) a seleccionar
        
        Resultado:
                  ?: Si j es int, devuelve la j-ésima fila del BlockM.
            Sistema: Si j es list, tuple o slice devuelve el BlockM cuyas
                  filas son las filas indicadas en la lista, tupla o slice de
                  índices.
        
        Ejemplos:
        >>> # Extrae la j-ésima fila del BlockM 
        >>> 1 | BlockM([['Hola', 2, 3], [1, 2, 5], [1, 2, 7]])
        
        BlockV(['Hola', 2, 3])
        >>> # Sistema formado por los elementos indicados en la lista (o tupla)
        >>> [2,1] | BlockM([['Hola', 2, 3], [1, 2, 5], [1, 2, 7]])
        >>> (2,1) | BlockM([['Hola', 2, 3], [1, 2, 5], [1, 2, 7]])
        
        BlockM([BlockV([1, 'Hola']), Vector([2, 2]), Vector([5, 3])])
        
        >>> # Sistema formado por los elementos indicados en el slice
        >>> slice(1,3,2) | BlockM([['Hola', 2, 3], [1, 2, 5], [1, 2, 7]]
        
        BlockM([BlockV(['Hola', 1]), Vector([2, 2]), Vector([3, 7])])
        
        """
        if isinstance(i,int):
            return  BlockV( (~self)|i , rpr='fila' )
    
        elif isinstance(i, (list,tuple,slice)):        
            return ~BlockM( (~self)|i ) 
            
       
    def vec(self):
        "Vectoriza un BlockM apilando sus elementos para formar un BlockV"
        return BlockV(Sistema([]).junta(self))
    
    
    def __rand__(self, operaciones):
        """Transforma las filas de un BlockM
        
        Atributos:
            operaciones (T): transformaciones a aplicar sobre las filas
                             de un BlockM A
        Ejemplos:
        >>>  T({1,3})   & A               # Intercambia las filas 1 y 3
        >>>  T((5,1))   & A               # Multiplica por 5 la fila 1
        >>>  T((5,2,1)) & A               # Suma 5 veces la fila 2 a la fila 1
        >>>  T([(5,2,1),(5,1),{1,3}]) & A # Aplica la secuencia de transformac.
                    # sobre las filas de A y en el orden inverso al de la lista
        
        """
        for item in reversed(operaciones.abreviaturas):
            if isinstance(item, (set, tuple) ):
                self.lista = (~(~self & T(item))).lista.copy()
        
            elif isinstance(item, list):
                for k in item:          
                    ~T(k) & self
        
        return self 
    
    
    def cfil(self, conjuntoIndices={}):
        """Modifica el atributo .corteElementos para insertar lineas
        horizontales entre las filas del BlockM
    
        """
        self.corteElementos = set(conjuntoIndices) if conjuntoIndices else {0}
        return self
    
       
    def extDiag(self, lista, c=False):
        "Extiende una BlockM a lo largo de la diagonal con una lista de BlockMs"
        lista = CreaLista(lista)
        if not all(isinstance(elemento, BlockM) for elemento in lista): 
            return ValueError('No es una lista de BlockMs')
        Ext_dos = lambda x, y: x.apila(M0(y.m,x.n),c).concatena(M0(x.m,y.n).apila(y,c),c)
        ExtDiag = lambda x: x[0] if len(x)==1 else Ext_dos( ExtDiag(x[0:-1]), x[-1] )
        return ExtDiag([self]+lista)
    
    
    def __repr__(self):
        """ Muestra un BlockM en su representación Python repr """
        return 'BlockM(' + repr(self.lista) + ')'
                               
    
    def __str__(self):
        """ Muestra un BlockM en su representación Python str """
        ln  = [len(n) for n in particion(self.corteSistema,self.n)]
        car = max([len(str(e)) for c in self for e in c])
    
        def escribeFila(f,d=0):
            parte = lambda f,d=0: str(' '.join([str(e).rjust(d) for e in f])) 
            s = '|'+'|'.join([parte([e for e in c],d) for c in [p|f for p in particion(self.corteSistema, self.n)] ])+'|'
            return s
        
        num_guiones = len(escribeFila(1|self, car))
        s = ('\n'+ '-'*(num_guiones) + '\n').join(['\n'.join([escribeFila(f,car) for f in ~s]) \
                                                   for s in [i|self for i in particion(self.corteElementos, self.m)]])
        return s
    
    def latex(self):
        """ Construye el comando LaTeX para representar una BlockM """
        ln = [len(n) for n in particion(self.corteSistema, self.n)]                                                           
        return \
            '\\left[ \\begin{array}{' + '|'.join([n*'c' for n in ln])  + '}' + \
            '\\\\ \\hline '.join(['\\\\'.join(['&'.join([latex(e) for e in f.lista]) \
                                               for f in (~M).lista]) \
               for M in [ i|self for i in particion(self.corteElementos,self.m)]]) + \
             '\\\\ \\end{array} \\right]'
            
class Matrix(BlockM):
    """Matrix un Sistema de Vectores con el mismo número de componentes.
    
    Una Matrix se puede instanciar con:
    
     1. una lista, tupla o Sistema de Vectores con el mismo número de
        componentes o longitud (serán las columnas).
     2. una lista, tupla o Sistema de listas, tuplas o Sistemas de núemros
        con la misma longitud (serán las filas de la matriz).
    
    Parámetros:
        arg (list, tuple, Sistema): Lista, tupla o Sistema de Vectores con
            mismo núm. de componentes (sus columnas); o de listas, tuplas
            o Sistemas de números de misma longitud (sus filas).
    
    Atributos heredados de la clase Sistema:
        lista              (list): list con los elementos.
        n                  (int) : número de elementos de la lista.
        corteSistema (set) : Conjunto de índices donde pintar
                                    separaciones visuales
    
    Atributos heredados de la subclase BlockM:
        m              (int) : número de filas
        corteElementos (set) : Conjunto de índices donde pintar
                                separaciones visuales entre filas
    
    Ejemplos:
    >>> # Crear una Matrix a partir de una lista de Vectores:
    >>> a = Vector( [1,2] ); b = Vector( [1,0] ); c = Vector( [9,2] )
    >>> Matrix( [a,b,c] )
    
    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ])
    >>> # Crear una Matrix a partir de una lista de listas de números
    >>> A = Matrix( [ [1,1,9], [2,0,2] ] )
    
    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ])
    >>> # Crea una Matrix a partir de otra Matrix
    >>> Matrix( A )
    
    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ])
    
    """
    
    def __init__(self, data):
        """Inicializa una Matrix con una
    
        1) lista, tupla o Sistema de Vectores con el mismo número de
        elementos,
        
        2) tupla, lista o Sistema de tuplas, listas o Sistemas de números
        con el mismo número de elementos,
        
        3) con otra Matrix.
    
        """
    
        super().__init__(data)
        
        lista = Sistema(data).lista
    
        if all([(isinstance(elemento, Vector) and len(elemento)==len(lista[0])) for elemento in lista]):
            self.lista   = lista.copy()
    
        elif Sistema(lista).es_de_composicion_y_longitud_uniforme() and es_ristra(lista[0]) and es_numero(lista[0][0]):
            self.lista = Matrix([ Vector([elemento[i] for elemento in lista]) for i in range(len(lista[0])) ]).lista
    
        elif isinstance(data,Matrix):
            self.lista   = data.lista.copy()
    
        else: 
            raise ValueError("""El argumento debe ser una lista de Vectores o una lista de listas o
            tuplas con el mismo número de elementos!""")
        
        super().__init__(data)
    
        for v in self.lista:
            v.rpr='columna'
            
    
    def es_cuadrada(self):
        """Indica si es cierto que la Matrix es cuadrada"""
        return self.m==self.n
        
    def es_simetrica(self):
        """Indica si es cierto que la Matrix es simétrica"""
        return self == ~self
        
    def es_triangularSup(self):
        """Indica si es cierto que la Matrix es triangular superior"""
        return not any(sum([[i|self|j for i in range(j+1,self.m+1)]      \
                                      for j in range(1  ,self.n+1)], []))
        
    def es_triangularInf(self):
        """Indica si es cierto que la Matrix es triangular inferior"""
        return (~self).es_triangularSup()
        
    def es_triangular(self):
        """Indica si es cierto que la Matrix es triangular inferior"""
        return self.es_triangularSup() | self.es_triangularInf()
        
    def es_diagonal(self):
        """Indica si es cierto que la Matrix es diagonal"""
        return self.es_triangularSup() & self.es_triangularInf()
    
    
    def diag(self):
        """Crea un Vector a partir de la diagonal de la Matriz"""
        return Vector([ (j|self|j) for j in range(1,min(self.m,self.n)+1)])
    
    
    def normalizada(self, opcion='Columnas'):
        if opcion == 'Columnas':
            if any( vector.es_nulo() for vector in self):
                raise ValueError('algún vector es nulo')
            return Matrix([ vector.normalizado() for vector in self])
        else:
            return ~(~self.normalizada())
            
    
    def __pow__(self,n):
        """Calcula la n-ésima potencia de una Matrix"""
        if not isinstance(n,int): raise ValueError('La potencia no es un entero')
        if not self.es_cuadrada:  raise ValueError('Matrix no es cuadrada')
    
        M = self if n else I(self.n)
        for i in range(1,abs(n)):
        	M = M * self
    
        return M.inversa() if n < 0 else M
    
    
    def det(self, sust=[]):
        """Calculo del determinate mediante la expansión de Laplace"""
        if not self.es_cuadrada(): raise ValueError('Matrix no cuadrada')
                                                                   
        def cof(self,f,c):
            """Cofactor de la fila f y columna c"""
            excl = lambda k: tuple(i for i in range(1,self.m+1) if i!=k)
            return (-1)**(f+c)*(excl(f)|self|excl(c)).det()
                                                                   
        if self.m == 1:
            return 1|self|1
    
        A = Matrix(self.subs(sust))
        # expansión por la 1ª columna 
        return sympy.simplify(sum([((f|A|1)*cof(A,f,1)).subs(sust) for f in range(1,A.m+1)])) 
                                                                                                                                  
    
    def GS(self):
        """Devuelve una Matrix equivalente cuyas columnas son ortogonales
    
        Emplea el método de Gram-Schmidt"""
        A = Matrix(self)
        for n in range(2,A.n+1):
            A & T([ (-fracc((A|n)*(A|j),(A|j)*(A|j)), j, n) \
                    for j in range(1,n) if (A|j).no_es_nulo() ])
        return A
    
    
    def rg(self):
        """Rango de una Matrix"""
        return [v.no_es_nulo() for v in self.K()].count(True)
    
    
    def es_singular(self):
        if not self.es_cuadrada():
            raise ValueError('La matriz no es cuadrada')
        return self.rg()<self.n
      
    def es_invertible(self):
        if not self.es_cuadrada():
            raise ValueError('La matriz no es cuadrada')
        return self.rg()==self.n
      
    
    def inversa(self, rep=0):                                                               
        """Inversa de Matrix"""
        if not self.es_cuadrada():
            raise ValueError('Matrix no cuadrada')
        
        pasos = self.elim(2).elim(20).elim(10).pasos
        TrF = pasos[0]
        TrC = pasos[1]
        tex   = rprElim(self.apila(I(self.n),1),pasos)
        
        if rep:
            display(Math(tex))
    
        if self.es_singular():
            raise ValueError('Matrix es singular')
    
        InvMat = I(self.n) & T(TrC)
        InvMat.pasos = pasos
        InvMat.TrF   = TrF
        InvMat.TrC   = TrC
        InvMat.tex   = tex
        return InvMat
    
    
    def determinante(self, rep = False, sust = []):
        """Calculo del determinate mediante eliminación"""
        if not self.es_cuadrada(): raise ValueError('Matrix no cuadrada')
        
        return Determinante(self.subs(sust),rep).valor
                           
    def diagonalizaS(self, espectro, rep=False, sust=[]):
        """Diagonaliza por bloques triangulares una Matrix cuadrada
        
        Encuentra una matriz diagonal semejante mediante trasformaciones de
        sus columnas y las correspondientes transformaciones inversas espejo
        de las filas. Requiere una lista de autovalores (espectro), que deben
        aparecer en dicha lista tantas veces como sus respectivas
        multiplicidades algebraicas. Los autovalores aparecen en la diagonal
        principal de la matriz diagonal. El atributo S de dicha matriz
        diagonal es una matriz cuyas columnas son autovectores de los
        correspondientes autovalores.  """
        D            = Matrix(self).copy().subs(sust)
        
        def no_son_autovalores(A, L):
            no_son=[l for i,l in enumerate(L) if (D-l*I(D.n)).es_invertible()]
            if no_son:
                print('los valores de la siguiente lista no son autovalores de la matriz:', no_son)
                return True
            else:
                False
        
        def espectro_correcto(A, L):
            x = sympy.symbols('x')    
            monomio = lambda l,x: l-x
            p = (A-x*I(A.n)).det()
            for l in L:
                p, r = sympy.div(p, monomio(l,x), domain ='QQ')    
            return False if p!=1 or r!=0 else True
        
        if not D.es_cuadrada:
            raise ValueError('Matrix no es cuadrada')
        
        if not isinstance(espectro, list):
            raise ValueError('espectro no es una lista')
        
        if no_son_autovalores(D, espectro):
            raise ValueError('quite de la lista los valores que no son autovalores')
            
        if not len(espectro)==D.n:
            raise ValueError('el espectro propocionado tiene un número inadecuado de autovalores')
        
        if not espectro_correcto(D, espectro):
            raise ValueError('introduzca una lista correcta de autovalores')
        
        espectro     = [sympy.S(l).subs(sust) for l in espectro]    
        S            = I(D.n)
        Tex          = latex( D.apila(S,1) )
        pasosPrevios = [[],[]]
        selecc       = list(range(1, D.n+1))
    
        for landa in espectro:
            m = selecc.pop()
            
            D = (D-(landa*I(D.n))).subs(sust)
            Tex += r'\xrightarrow[\boxed{' + latex(landa) + r'\mathbf{I}}]{(-)}' + latex((D.apila(S,1)).subs(sust))
            
            # eliminamos elementos superiores de la columna con elim de izda a dcha
            TrCol = filtradopasos((slice(None,m)|D|slice(None,m)).elim(20).pasos[1])
            
            if T(TrCol):
                Tex             = rprElim( D.apila(S,1),  [[], TrCol], Tex, sust) if TrCol else Tex
                D               = (D & T(TrCol)).subs(sust)
                S               = (S & T(TrCol)).subs(sust)
                pasosPrevios[1] = pasosPrevios[1] + TrCol
                
                TrFilas         = [ T( [op.Tinversa().espejo() for op in TrCol[::-1]] ) ]
                
                Tex             = rprElim( D.apila(S,1), [TrFilas, []], Tex, sust) if TrCol else Tex
                D               = (T(TrFilas) & D).subs(sust)
                pasosPrevios[0] = TrFilas + pasosPrevios[0]
            
            
            if m < D.n: # eliminamos elementos inferiores de la columna con los pivotes de la diagonal
                for i,_ in enumerate(slice(m+1,None)|D|m, m+1):
                    TrCol = filtradopasos([ T([(-fracc(i|D|m, i|D|i), i, m)]) ]) if i|D|i else [T([])]
                    
                    if T(TrCol):
                        Tex             = rprElim( D.apila(S,1),  [[], TrCol], Tex, sust) if TrCol else Tex
                        D               = (D & T(TrCol)).subs(sust)
                        S               = (S & T(TrCol)).subs(sust)
                        pasosPrevios[1] = pasosPrevios[1] + TrCol
                        
                        TrFilas         = [ T( [op.Tinversa().espejo() for op in TrCol[::-1]] ) ]
                        
                        Tex             = rprElim( D.apila(S,1), [TrFilas, []], Tex, sust) if TrCol else Tex
                        D               = (T(TrFilas) & D).subs(sust)
                        pasosPrevios[0] = TrFilas + pasosPrevios[0]
                         
            
            D = (D+(landa*I(D.n))).subs(sust)
            Tex += r'\xrightarrow[\boxed{' + latex(landa) + r'\mathbf{I}}]{(+)}' + latex((D.apila(S,1)).subs(sust))
            
                        
        if rep: display(Math(Tex))
        D.espectro = espectro[::-1]
        D.tex = Tex
        D.S   = S
        D.TrC = pasosPrevios[1]
        D.TrF = [op.Tinversa().espejo() for op in D.TrC[::-1]]
        D.pasos = [D.TrF, D.TrC]
        return D
        
    def diagonalizaC(self, rep=False, sust=[], variante=0):
        """Diagonaliza por congruencia una Matrix simétrica (evitando dividir)
        
        Encuentra una matriz diagonal por conruencia empleando una matriz B
        invertible (evitando fracciones por defecto, si variante=1 entonces no
        evita las fracciones) por la derecha y su transpuesta por la
        izquierda. No emplea los autovalores. En general los elementos en la
        diagonal principal no son autovalores, pero hay tantos elementos
        positivos en la diagonal como autovalores positivos, tantos negativos
        como autovalores negativos, y tantos ceros como auntovalores nulos.
        
        """
        
        def BuscaPrimerNoNuloEnLaDiagonal(self, i=1):
            """Indica el índice de la primera componente no nula de a diagonal
            desde de la posición i en adelante. Si son todas nulas devuelve 0
        
            """
            
            d = (slice(i,None)|self|slice(i,None)).diag().sis()
            return next((pos for pos, x in enumerate(d) if x), -i) + i
        
        if not variante in {0,1}:
            raise ValueError('La variante debe ser 0 ó 1')
        
        D            = Matrix(self).copy().subs(sust)
        
        if not D.es_simetrica():
            raise ValueError('La matriz no es simétrica')
         
        pasosPrevios = [ [], [] ]
        
        for i in range(1, D.n):
            p = BuscaPrimerNoNuloEnLaDiagonal(D, i)
            j = [k for k,col in enumerate(D|slice(i,None),i) if (i|col and not k|col)]
            
            if not (i|D|i):
                if p:
                    Tr = T( {i, p} )
                    p = i
                    
                    pasos = [ filtradopasos([~Tr]), filtradopasos([Tr]) ]
                    pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                    pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                    D = (T(pasos[0]) & D & T(pasos[1])).subs(sust)
                    
                elif j:
                    Tr = T( (1, j[0], i) )
                    p = i
                    
                    pasos = [ filtradopasos([~Tr]), filtradopasos([Tr]) ]
                    pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                    pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                    D = (T(pasos[0]) & D & T(pasos[1])).subs(sust)
                    
            if p:
                Tr = T(((i,)|D).elim(variante).pasos[1]) 
                
                pasos = [ filtradopasos([~Tr]), filtradopasos([Tr]) ]
                pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                D = (T(pasos[0]) & D & T(pasos[1])).subs(sust)
                
               
        D.pasos     = pasosPrevios
        D.tex       = rprElimCF(Matrix(self).subs(sust), D.pasos, [], sust) 
        D.TrF       = filtradopasos(T(D.pasos[0]))
        D.TrC       = filtradopasos(T(D.pasos[1]))
        D.B         = I(self.n) & D.TrC
        
        if rep: 
            display(Math(D.tex))
    
        return D   
        
    
    def diagonalizaO(self, espectro, sust=[]):
        """Diagonaliza ortogonalmente una Matrix simétrica
        
        Encuentra una matriz diagonal por semejanza empleando una matriz
        ortogonal Q a la derecha y su inversa (transpuesta) por la izquierda.
        Requiere una lista de autovalores (espectro), que deben aparecer
        tantas veces como sus respectivas multiplicidades algebraicas. Los
        autovalores aparecen en la diagonal principal de la matriz
        diagonal. El atributo Q de la matriz diagonal es la matriz ortogonal
        cuyas columnas son autovectores de los correspondientes autovalores.
        
        """
        D            = Matrix(self).copy().subs(sust)
        
        if not D.es_simetrica():
            raise ValueError('La matriz no es simétrica')
        
        
        def no_son_autovalores(A, L):
            no_son=[l for i,l in enumerate(L) if (D-l*I(D.n)).es_invertible()]
            if no_son:
                print('los valores de la siguiente lista no son autovalores de la matriz:', no_son)
                return True
            else:
                False
        
        def espectro_correcto(A, L):
            x = sympy.symbols('x')    
            monomio = lambda l,x: l-x
            p = (A-x*I(A.n)).det()
            for l in L:
                p, r = sympy.div(p, monomio(l,x), domain ='QQ')    
            return False if p!=1 or r!=0 else True
        
        if not D.es_cuadrada:
            raise ValueError('Matrix no es cuadrada')
        
        if not isinstance(espectro, list):
            raise ValueError('espectro no es una lista')
        
        if no_son_autovalores(D, espectro):
            raise ValueError('quite de la lista los valores que no son autovalores')
            
        if not len(espectro)==D.n:
            raise ValueError('el espectro propocionado tiene un número inadecuado de autovalores')
        
        if not espectro_correcto(D, espectro):
            raise ValueError('introduzca una lista correcta de autovalores')
        
        
        def BaseOrtNor(q):
            "Crea una base ortonormal cuyo último vector es 'q'"
            if not isinstance(q,Vector): raise ValueError('El argumento debe ser un Vector')
            M = Matrix([q]).concatena(I(q.n)).GS()
            l = [ j for j, v in enumerate(M, 1) if v.no_es_nulo() ]
            l = l[1:len(l)]+[l[0]]
            return (M|l).normalizada()
        
        
        espectro     = [sympy.S(l).subs(sust) for l in espectro]    
        S            = I(D.n)
        Tex          = latex( D.apila(S,1) )
        pasosPrevios = [[],[]]
        selecc       = list(range(1,D.n+1))
    
        for l in espectro:
            D       = (D - l*I(D.n)).subs(sust)
            k       = len(selecc)
            nmenosk = (D.n)-k
            m       = selecc.pop()
            TrCol   = filtradopasos((slice(None,m)|D|slice(None,m)).elim(20, False, sust).pasos[1])
            D       = (D + l*I(D.n)).subs(sust)
            
            
            q = ( (I(k) & T(TrCol)).subs(sust) )|0
            q = (sympy.sqrt(q*q)) * q
            
            Q = BaseOrtNor(q).concatena(M0(k,nmenosk)).apila( \
                    M0(nmenosk,k).concatena(I(nmenosk)))  if nmenosk else BaseOrtNor(q)
                
            S = (S*Q).subs(sust)
            D = (~Q*D*Q).subs(sust)
    
        D.Q = S
        D.espectro = espectro[::-1]
        return D
    
    
    def __repr__(self):
        """ Muestra una Matrix en su representación Python repr """
        return 'Matrix(' + repr(self.lista) + ')'
                               

class M0(Matrix):
    def __init__(self, m, n=None):
        """ Inicializa una matriz nula de orden m por n """
        n = m if n is None else n

        super().__init__( [V0(m)]*n )
        self.__class__ = Matrix

class M1(Matrix):
    def __init__(self, m, n=None):
        """ Inicializa una matriz nula de orden m por n """
        n = m if n is None else n

        super().__init__( [V1(m)]*n )
        self.__class__ = Matrix

class I(Matrix):
    def __init__(self, n):
        """ Inicializa la matriz identidad de tamaño n """
        super().__init__([[(i==j)*1 for i in range(n)] for j in range(n)])
        self.__class__ = Matrix



class T:
    """Clase para las transformaciones elementales
    
    T ("Transformación elemental") guarda en su atributo 'abreviaturas'
    una abreviatura (o una secuencia de abreviaturas) de transformaciones
    elementales. El método __and__ actúa sobre otra T para crear una T que
    es composición de transformaciones elementales (una la lista de
    abreviaturas), o bien actúa sobre una BlockM (para transformar sus
    filas).
    
    Una T (transformación elemental) se puede instanciar indicando las
    operaciones mediante un número arbitrario de
    
     1. abreviaturas(set): {índice, índice}. Abrev. de un intercambio de
                           entre los elementos correspondientes a dichos
                           índices
    
                  (tuple): (escalar, índice). Abrev. transf. Tipo II que
                           multiplica el elemento correspondiente al
                           índice por el escalar
    
                           (escalar, índice1, índice2). Abrev.
                           transformación Tipo I que suma al elemento
                           correspondiente al índice2 el elemento
                           correspondiente al índice1 multiplicado por
                           el escalar
    
     2. transf. Elems.(T): Genera otra T cuyo atributo .abreviaturas es
                           una copia del atributo .abreviaturas de la
                           transformación dada
    
     3.      listas(list): Con cualesquiera de los anteriores objetos o
                           con sublistas formadas con los anteriores
                           objetos. Genera una T cuyo atributo
                           .abreviaturas es una concatenación de todas las
                           abreviaturas
    
    
    Atributos:
    
       abreviaturas (set): lista con las abreviaturas de todas las
                           transformaciones
    
                rpr (str): Si rpr='v' (valor por defecto) se muestra la
                           lista de breviaturas en vertical. Con cualquier
                           otro valor se muestran en horizontal.
    
    Ejemplos:
    >>> # Intercambio entre elementoes
    >>> T( {1,2} )
    
    >>> # Trasformación Tipo II (multiplica por 5 el segundo elemento)
    >>> T( (5,2) )
    
    >>> # Trasformación Tipo I (resta el tercer elemento al primero)
    >>> T( (-1,3,1) )
    
    >>> # Secuencia de las tres transformaciones anteriores
    >>> T( [{1,2}, (5,2), (-1,3,1)] )
    
    >>> # T de una T
    >>> T( T( (5,2) ) )
    
    T( (5,2) )
    
    >>> # T de una lista de T's
    >>> T( [T([(-8, 2), (2, 1, 2)]), T([(-8, 3), (3, 1, 3)]) ] )
    
    T( [(-8, 2), (2, 1, 2), (-8, 3), (3, 1, 3)] )
    
    """
    
    def __init__(self, *args, rpr='v'):
        """Inicializa una transformación elemental"""
        
        def verificacion(abrv):
            if isinstance(abrv,tuple) and (len(abrv) == 2) and abrv[0]==0:
                raise ValueError('T( (0, i) ) no es una trasformación elemental')
            if isinstance(abrv,tuple) and (len(abrv) == 3) and (abrv[1] == abrv[2]):
                raise ValueError('T( (a, i, i) ) no es una trasformación elemental')
            if isinstance(abrv,set) and len(abrv)>2:
                raise ValueError ('El conjunto debe tener uno o dos índices para ser un intercambio')
            
        
        def simplificacionSimbolica(arg):
            if isinstance(arg,tuple) and (len(arg) == 2):
                return (sympy.factor(arg[0]), arg[1],)
            elif isinstance(arg,tuple) and (len(arg) == 3):
                return (sympy.factor(arg[0]), arg[1], arg[2],)
            else:
                return arg
        
        def CreaListaAbreviaturas(arg):
            if isinstance(arg, (tuple, set)):
                verificacion(arg)
                arg = simplificacionSimbolica(arg)
                return [arg]
            if isinstance(arg, list):
                return [abrv for item in arg for abrv in CreaListaAbreviaturas(item)]
            if isinstance(arg, T):
                return CreaListaAbreviaturas(arg.abreviaturas) 
        
        def concatenaTodasLasAbreviaturasDeLos(args):
            return [abrv for item in args for abrv in CreaListaAbreviaturas(item)]
        
        self.abreviaturas = concatenaTodasLasAbreviaturasDeLos(args)
        self.rpr          = rpr
    
    
    def __and__(self, other):
        """Composición de transformaciones elementales (o transformación filas)
        
        Crea una T con una lista de abreviaturas de transformaciones elementales
        (o llama al método que modifica las filas de una Matrix)
        
        Parámetros:
            (T): Crea la abreviatura de la composición de transformaciones, es
                 decir, una lista de abreviaturas
            (Matrix): Llama al método de la clase Matrix que modifica sus filas
        
        Ejemplos:
        >>> # Composición de dos Transformaciones elementales
        >>> T( {1, 2} ) & T( (2, 4) )
        
        T( [{1,2}, (2,4)] )
        
        >>> # Composición de dos Transformaciones elementales
        >>> T( {1, 2} ) & T( [(2, 4), (2, 1), {3, 1}] )
        
        T( [{1, 2}, (2, 4), (2, 1), {3, 1}] )
        
        >>> # Transformación de las filas de una Matrix
        >>> T( [{1,2}, (4,2)] ) & A # multiplica por 4 la segunda fila de A y
                                    # luego intercambia las dos primeras filas
        """        
        
        if isinstance(other, T):
            return T(self.abreviaturas+other.abreviaturas, rpr=self.rpr)
    
        if isinstance(other, Sistema):
            return other.__rand__(self)
    
    
    def __invert__(self):
        """Transpone la lista de abreviaturas (invierte su orden)"""
        return T( list(reversed(self.abreviaturas)), rpr=self.rpr) if isinstance(self.abreviaturas, list) else self
        
    
    def Tinversa ( self ):
        """Calculo de la inversa de una transformación elemental"""
        operaciones = [                      abrv     if isinstance(abrv,set) else \
                        ( -abrv[0], abrv[1], abrv[2]) if len(abrv)==3         else \
                        (fracc(1,abrv[0]),   abrv[1])   for abrv in CreaLista(self.abreviaturas) ]
    
        return ~T( operaciones, rpr=self.rpr)
    
    
    def __pow__(self,n):
        """Calcula potencias de una T (incluida la inversa)"""
        
        def Tinversa ( self ):
            """Calculo de la inversa de una transformación elemental"""
            operaciones = [                      abrv     if isinstance(abrv,set) else \
                            ( -abrv[0], abrv[1], abrv[2]) if len(abrv)==3         else \
                            (fracc(1,abrv[0]),   abrv[1])   for abrv in CreaLista(self.abreviaturas) ]
        
            return ~T( operaciones, rpr=self.rpr)
            
        if not isinstance(n,int):
            raise ValueError('La potencia no es un entero')
    
        potencia = lambda T, n: T if n==1 else T & potencia(T, n-1)
        TransformacionElemental = potencia(self,abs(n)) if n!=0  else  T({1})
        
        return TransformacionElemental if n>0 else Tinversa(TransformacionElemental)
            
    def espejo ( self ):
        """Calculo de la transformación elemental espejo de otra"""
        return T([ (abrv[0], abrv[2], abrv[1]) if len(abrv)==3 else abrv for abrv in CreaLista(self.abreviaturas)], rpr=self.rpr)
        
    
    def subs(self, regla_de_sustitucion=[]):
        '''Sustitución simbólica en transformaciones elementales'''
        
        def sustitucion(operacion, regla_de_sustitucion):
            if isinstance(operacion, tuple):
                return tuple(sympy.S(operacion).subs(CreaLista(regla_de_sustitucion)) )
            elif isinstance(operacion, set):
                return set(sympy.S(operacion).subs(CreaLista(regla_de_sustitucion)) )
            elif isinstance(operacion, list):
                return [sustitucion(item, regla_de_sustitucion) for item in operacion] 
            elif isinstance(operacion, T):
                return operacion.subs(CreaLista(regla_de_sustitucion)) 
            
        self = T([sustitucion(operacion, regla_de_sustitucion) for operacion in self.abreviaturas])
        return self
    
    def simplify(arg):
        if isinstance(arg,tuple) and (len(arg) == 2):
            return (simplify(arg[0]), arg[1],)
        elif isinstance(arg,tuple) and (len(arg) == 3):
            return (simplify(arg[0]), arg[1], arg[2],)
        else:
            return arg
    
    def factor(arg):
        if isinstance(arg,tuple) and (len(arg) == 2):
            return (factor(arg[0]), arg[1],)
        elif isinstance(arg,tuple) and (len(arg) == 3):
            return (factor(arg[0]), arg[1], arg[2],)
        else:
            return arg
    
    def expand(arg):
        if isinstance(arg,tuple) and (len(arg) == 2):
            return (expand(arg[0]), arg[1],)
        elif isinstance(arg,tuple) and (len(arg) == 3):
            return (expand(arg[0]), arg[1], arg[2],)
        else:
            return arg
    
    
    def __getitem__(self,i):
        """ Devuelve las transformaciones elementales del i-ésimo paso """
        return T(self.abreviaturas[i])
    
    def __setitem__(self,i,value):
        """ Modifica las transformaciones elementales del i-ésimo paso """
        self.abreviaturas[i]=value
            
    def __len__(self):
        """Número de pasos de T """
        return len(self.abreviaturas)
    
    
    def __eq__(self, other):
        """Indica si es cierto que dos Transformaciones elementales son iguales"""
        return self.abreviaturas == other.abreviaturas
    
    
    def __repr__(self):
        """ Muestra T en su representación Python """
        return 'T(' + repr(self.abreviaturas) + ')'
    
    def _repr_html_(self):
        """ Construye la representación para el entorno Jupyter Notebook """
        return html(self.latex())
    
    def latex(self):
        """ Construye el comando LaTeX para representar una Trans. Elem. """
        def simbolo(t):
            """Escribe el símbolo que denota una trasformación elemental particular"""
            if isinstance(t,(set,sympy.sets.sets.FiniteSet)):
                return '\\left[\\mathbf{' + latex(list(t)[0]) + \
                  '}\\rightleftharpoons\\mathbf{' + latex(list(t)[-1]) + '}\\right]'
            if isinstance(t,(tuple, sympy.core.containers.Tuple)) and len(t) == 2:
                return '\\left[\\left(' + \
                  latex(t[0]) + '\\right)\\mathbf{'+ latex(t[1]) + '}\\right]'
            if isinstance(t,(tuple, sympy.core.containers.Tuple)) and len(t) == 3:
                return '\\left[\\left(' + latex(t[0]) + '\\right)\\mathbf{' + \
                  latex(t[1]) + '}' + '+\\mathbf{' + latex(t[2]) + '} \\right]'    
    
        if isinstance(self.abreviaturas, (set, tuple) ):
            return '\\underset{' + simbolo(self.abreviaturas) + '}{\\pmb{\\tau}}'
    
        elif self.abreviaturas == []:
            return ' '
    
        elif isinstance(self.abreviaturas, list) and self.rpr=='v':
            return '\\underset{\\begin{subarray}{c} ' + \
                  '\\\\'.join([simbolo(i) for i in self.abreviaturas])  + \
                  '\\end{subarray}}{\\pmb{\\tau}}'
    
        elif isinstance(self.abreviaturas, list):
            return '\\underset{' + \
                   '}{\\pmb{\\tau}}\\underset{'.join([simbolo(i) for i in self.abreviaturas]) + \
                   '}{\\pmb{\\tau}}'
                  


class Elim(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve una forma pre-escalonada de un sistema

           operando con sus elementos (y evitando operar con
           fracciones).  Si rep es no nulo, se muestran en Jupyter los
           pasos dados

        """
        self.__dict__.update(sistema.K(rep, sust, repsust).__dict__)
        self.__class__ = type(sistema)


class ElimG(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve una forma escalonada de un sistema

           operando con sus elementos (y evitando operar con
           fracciones).  Si rep es no nulo, se muestran en Jupyter los
           pasos dados

        """
        self.__dict__.update(sistema.L(rep, sust, repsust).__dict__)
        self.__class__ = type(sistema)


class ElimGJ(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve la forma escalonada reducida de un sistema

           operando con sus elementos (y evitando operar con
           fracciones).  Si rep es no nulo, se muestran en Jupyter los
           pasos dados

        """
        self.__dict__.update(sistema.R(rep, sust, repsust).__dict__)
        self.__class__ = type(sistema)


class ElimGF(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve la forma escalonada por filas (si es posible)

           y evitando operar con fracciones.  Si rep es no nulo, se
           muestran en Jupyter los pasos dados

        """        
        self.__dict__.update(sistema.U(rep, sust, repsust).__dict__)
        self.__class__ = type(sistema)


class ElimGJF(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve la forma escalonada reducida por filas (si es posible)

           y evitando operar con fracciones.  Si rep es no nulo, se
           muestran en Jupyter los pasos dados

        """        
        self.__dict__.update(sistema.UR(rep, sust, repsust).__dict__)
        self.__class__ = type(sistema)


class InvMat(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve la matriz inversa y los pasos dados sobre las columnas

           y evitando operar con fracciones.  Si rep es no nulo, se
           muestran en Jupyter los pasos dados

        """
        
        def texYpasos(data, pasos, rep=0, sust=[], repsust=0):
            pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
            TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
            if repsust:
                tex = rprElim(data, pasos, TexPasosPrev, sust)
            else:
                tex = rprElim(data, pasos, TexPasosPrev)
            pasos[0] = pasos[0] + pasosPrevios[0] 
            pasos[1] = pasosPrevios[1] + pasos[1]
            
            if rep:
                display(Math(tex))
            
            return tex, pasos
        
        A = sistema.subs(sust).inversa()
        A.tex, A.pasos = texYpasos(sistema.apila(I(sistema.n),1), A.pasos, rep, sust, repsust)
        A.TrF = A.pasos[0]
        A.TrC = A.pasos[1]
        self.__dict__.update(A.__dict__)
        self.__class__ = type(sistema)


class InvMatF(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas

           y evitando operar con fracciones.  Si rep es no nulo, se
           muestran en Jupyter los pasos dados

        """
        
        def texYpasos(data, pasos, rep=0, sust=[], repsust=0):
            pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
            TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
            if repsust:
                tex = rprElim(data, pasos, TexPasosPrev, sust)
            else:
                tex = rprElim(data, pasos, TexPasosPrev)
            pasos[0] = pasos[0] + pasosPrevios[0] 
            pasos[1] = pasosPrevios[1] + pasos[1]
            
            if rep:
                display(Math(tex))
            
            return tex, pasos
        

        if not sistema.es_cuadrada():
            raise ValueError('Matrix no cuadrada')
    
        pasos = sistema.elim(26).elim(26).elim(14).pasos
        A = T(pasos[0]) & I(sistema.n)
        A.tex, A.pasos = texYpasos(sistema.concatena(I(sistema.n),1), pasos, rep, sust, repsust)
        A.TrF = A.pasos[0]
        A.TrC = A.pasos[1]
        self.__dict__.update(A.__dict__)
        self.__class__ = type(sistema)


class InvMatFC(Sistema):
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas y columnas

           y evitando operar con fracciones.  Si rep es no nulo, se
           muestran en Jupyter los pasos dados

        """
        
        def texYpasos(data, pasos, rep=0, sust=[], repsust=0):
            pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
            TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
            if repsust:
                tex = rprElim(data, pasos, TexPasosPrev, sust)
            else:
                tex = rprElim(data, pasos, TexPasosPrev)
            pasos[0] = pasos[0] + pasosPrevios[0] 
            pasos[1] = pasosPrevios[1] + pasos[1]
            
            if rep:
                display(Math(tex))
            
            return tex, pasos
        

        if not sistema.es_cuadrada():
            raise ValueError('Matrix no cuadrada')
    
        pasos = sistema.elim(24).elim(10).pasos
        A = sistema.copy()
        nan = sympy.symbols('\ ')
        dummyMatrix = M1(A.n)*nan
        A.tex, A.pasos = texYpasos(A.concatena(I(A.n),1).apila(I(A.n).concatena(dummyMatrix,1),1), pasos, rep, sust, repsust)
        A.TrF = A.pasos[0]
        A.TrC = A.pasos[1]
        A.F   = T(A.TrF) & I(A.n)
        A.C   = I(A.n) & T(A.TrC)
        A.lista = (A.C*A.F).lista
        self.__dict__.update(A.__dict__)
        self.__class__ = type(sistema)



class SubEspacio:
    def __init__(self, data, sust=[], Rn=[]):
        """Inicializa un SubEspacio de Rn"""
        
        def sistema_generador_del_espacio_nulo_de(A, sust=[], Rn=[]):
            """Encuentra un sistema generador del Espacio Nulo de A"""
            K = A.K(0, sust);
            E = I(A.n) & T(K.pasos[1])
            lista = [ v.reshape(Rn) for j, v in enumerate(E,1) if (K|j).es_nulo(sust) ]
            return Sistema(lista) if lista else Sistema([self.vector_nulo()])
        
        if not isinstance(data, Sistema):
            raise ValueError(' Argumento debe ser un Sistema o Matrix ')
        
        if not data:
            if not Rn:
                raise ValueError(' Si el sistema es vacio, es necesario indicar el espacio Rn ')
            else:
                self.Rn = Rn
                self.__dict__ = SubEspacio(Sistema([self.vector_nulo()])).__dict__.copy()
            
        elif isinstance(data, Matrix):
            A         = data
            self.Rn   = Rn if Rn else A.n
            self.sgen = sistema_generador_del_espacio_nulo_de(A, sust, Rn)
            self.dim  = 0 if self.sgen.es_nulo() else len(self.sgen)
            self.base = self.sgen if self.dim else Sistema([])
            self.cart = SubEspacio(self.sgen, sust, self.Rn).cart
    
        else:
            if isinstance(data|1, BlockM):
                self.Rn = ((data|1).m, (data|1).n)
            elif isinstance(data|1, Sistema):
                self.Rn =  (data|1).n
                
            try:
                A = Matrix(data).subs(sust)
            except:
                A = BlockM([data]).subs(sust)
            self.base = Sistema([data.K()|j for j,v in enumerate(A.elim(0), 1) if v.no_es_nulo()])
            self.dim  = len(self.base)
            self.sgen = self.base if self.base else Sistema([ self.vector_nulo() ])
            
            if isinstance(self.Rn, int):
                self.cart = ~Matrix(sistema_generador_del_espacio_nulo_de(~A, sust))
            elif isinstance(self.Rn, tuple):
                self.cart = SubEspacio(Sistema([m.vec() for m in self.sgen]), Rn=self.Rn).cart
        
    
    def vector_nulo(self):
        return M0(self.Rn[0],self.Rn[1]) if isinstance(self.Rn, tuple) else V0(self.Rn)
    
    
    def contenido_en(self, other):
        """Indica si este SubEspacio está contenido en other"""
        self.verificacion(other)
        if isinstance(other, SubEspacio):
            if isinstance(self.sgen|1, Vector):
                return all ([ (other.cart*v).es_nulo() for v in self.sgen ])
            elif isinstance(self.sgen|1, Matrix):
                return all ([ (other.cart*v.vec()).es_nulo() for v in self.sgen ])
            
        elif isinstance(other, EAfin):
            return other.v.es_nulo() and self.contenido_en(other.S)
    
    
    def __eq__(self, other):
        """Indica si un subespacio de Rn es igual a otro"""
        self.verificacion(other)
        return self.contenido_en(other) and other.contenido_en(self)
    
    def __ne__(self, other):
        """Indica si un subespacio de Rn es distinto de otro"""
        self.verificacion(other)
        return not (self == other)
    
    
    def verificacion(self,other):
        if not isinstance(other, (SubEspacio, EAfin)) or  not self.Rn == other.Rn: 
            raise \
             ValueError('Ambos argumentos deben ser subconjuntos de en un mismo espacio')
    
    
    def __add__(self, other):
        """Devuelve la suma de subespacios de Rn"""
        self.verificacion(other)
        return SubEspacio(self.sgen.concatena(other.sgen))
    
    
    def __and__(self, other):
        """Devuelve la intersección de subespacios"""
        self.verificacion(other)
        
        if isinstance(other, SubEspacio):
            A  = self.base
            B  = other.base
            AB = A.concatena(B)
            X  = slice(1,self.dim)|Matrix(AB.espacio_nulo().sgen)
            return SubEspacio(A*X)
        
        elif  isinstance(other, EAfin):
            return other & self
    
    
    def __invert__(self):
        """Devuelve el complemento ortogonal"""
        if isinstance(self.sgen|1, Vector):
            return SubEspacio( Sistema( ~(self.cart) ) )
        
        elif isinstance(self.sgen|1, Matrix):
            return SubEspacio(Sistema([v.reshape(self.Rn) for v in ~(self.cart)]))
        
    
    def __contains__(self, other):
        """Indica si un Vector pertenece a un SubEspacio"""
    
        if isinstance(self.sgen|1, Vector):
            if not isinstance(other, Vector) or other.n != self.Rn:
                raise ValueError\
                    ('El Vector no tiene el número adecuado de componentes')
            return self.cart*other == V0(self.cart.m)
        
        elif isinstance(self.sgen|1, Matrix):
            if not isinstance(other, Matrix) or (other.n,other.m) != self.Rn:
                raise ValueError\
                    ('El Vector no tiene el número adecuado de componentes')        
            return self.cart*other.vec() == V0(self.cart.m)
        
    
    
    def _repr_html_(self):
        """Construye la representación para el entorno Jupyter Notebook"""
        return html(self.latex())
    
    def EcParametricas(self, d=0):
        """Representación paramétrica del SubEspacio"""
        if d: display(Math(self.EcParametricas()))
        return EAfin(self.sgen, self.vector_nulo()).EcParametricas()
    
    def EcCartesianas(self, d=0):
        """Representación cartesiana del SubEspacio"""
        if d: display(Math(self.EcCartesianas()))
        return EAfin(self.sgen, self.vector_nulo()).EcCartesianas()
        
    def latex(self):
        """ Construye el comando LaTeX para un SubEspacio de Rn"""
        return EAfin(self.sgen, self.vector_nulo()).latex()       
    


class EAfin:
    
    def __init__(self, data, v, vi=0):
        """Inicializa un Espacio Afín de Rn"""
        self.S  = data if isinstance(data, SubEspacio) else SubEspacio(data)
        self.Rn = self.S.Rn
        
        if isinstance(self.Rn, int):
            if not isinstance(v, Vector) or v.n != self.Rn:
                raise ValueError('v y SubEspacio deben estar en el mismo espacio vectorial')
            
        elif isinstance(self.Rn, tuple):
            if not isinstance(v, Matrix) or (v.m,v.n) != self.Rn:
                raise ValueError('v y SubEspacio deben estar en el mismo espacio vectorial')
            
        self.v  = v if vi else (self.S.sgen.concatena(Sistema([v]))).elim(1)|0
        
    
    def __contains__(self, other):
        """Indica si un Vector pertenece a un EAfin"""
        if isinstance(self.S.sgen|1, Vector):
            if not isinstance(other, Vector) or other.n != self.Rn:
                raise ValueError\
                    ('El Vector no tiene el número adecuado de componentes')
            return self.S.cart*other == (self.S.cart)*self.v
        
        elif isinstance(self.S.sgen|1, Matrix):
            if not isinstance(other, Matrix) or (other.n,other.m) != self.Rn:
                raise ValueError\
                    ('La matrix no tiene el orden adecuado')        
            return self.S.cart*other.vec() == self.S.cart*self.v.vec()
    
    
    def contenido_en(self, other):
        """Indica si este EAfin está contenido en other"""
        self.verificacion(other)
        
        if isinstance(other, SubEspacio):
            return self.v in other and self.S.contenido_en(other)
        
        elif isinstance(other, EAfin):
             return self.v in other and self.S.contenido_en(other.S)
    
    
    def __eq__(self, other):
        """Indica si un EAfin de Rn es igual a other"""
        self.verificacion(other)
        return self.contenido_en(other) and other.contenido_en(self)
    
    def __ne__(self, other):
        """Indica si un subespacio de Rn es distinto de other"""
        return not (self == other)
    
    
    def verificacion(self,other):
        if not isinstance(other, (SubEspacio, EAfin)) or  not self.Rn == other.Rn: 
            raise ValueError('Ambos argumentos deben ser subconjuntos de en un mismo espacio')
    
    
    def __and__(self, other):
        """Devuelve la intersección de este EAfin con other"""
        self.verificacion(other)
        if isinstance(other, EAfin):
            M = self.S.cart.apila( other.S.cart )
            if isinstance(self.S.sgen|1, Vector):
                w = (self.S.cart*self.v).concatena( other.S.cart*other.v )
            elif isinstance(self.S.sgen|1, Matrix):
                w = (self.S.cart*self.v.vec()).concatena( other.S.cart*other.v.vec() )
        elif isinstance(other, SubEspacio):
            M = self.S.cart.apila( other.cart )
            if isinstance(self.S.sgen|1, Vector):
                w = (self.S.cart*self.v).concatena( V0(other.cart.m) )                                                      
            elif isinstance(self.S.sgen|1, Matrix):
                w = (self.S.cart*self.v.vec()).concatena( V0(other.cart.m) )
                
        return SEL(M,w).eafin
    
    
    def __invert__(self):
        """Devuelve el mayor SubEspacio perpendicular a self"""
        return SubEspacio( Sistema([v.reshape(self.Rn) for v in ~(self.S.cart)]))
    
    
    def _repr_html_(self):
        """Construye la representación para el entorno Jupyter Notebook"""
        return html(self.latex())
    
    def EcParametricas(self, d=0):
        """Representación paramétrica de EAfin"""
        punto = latex(self.v) + '+' if (self.v != 0*self.v) else ''
        if d: display(Math(self.EcParametricas()))
        if isinstance(self.S.Rn,int):
            return r'\left\{ \boldsymbol{v}\in\mathbb{R}^' \
                + latex(self.S.Rn)  \
                + r'\ \left|\ \exists\boldsymbol{p}\in\mathbb{R}^' \
                + latex(max(self.S.dim,1)) \
                + r',\; \boldsymbol{v}= ' \
                + punto \
                + latex(Matrix(self.S.sgen)) \
                + r'\boldsymbol{p}\right. \right\}' 
        else:
            return r'\left\{ \pmb{\mathsf{M}}\in\mathbb{R}^{' \
                + latex(self.S.Rn[0]) + r'\times' + latex(self.S.Rn[1]) + '}' \
                + r'\ \left|\ \exists\boldsymbol{p}\in\mathbb{R}^' \
                + latex(max(self.S.dim,1)) \
                + r',\; \mathsf{\pmb{M}}= ' \
                + punto \
                + latex(self.S.sgen) \
                + r'\boldsymbol{p}\right. \right\}' 
                        
    def EcCartesianas(self, d=0):
        """Representación cartesiana de EAfin"""
        if d: display(Math(self.EcCartesianas()))
        if isinstance(self.S.Rn,int):
            return r'\left\{ \boldsymbol{v}\in\mathbb{R}^' \
                + latex(self.S.Rn) \
                + r'\ \left|\ ' \
                + latex(self.S.cart) \
                + r'\boldsymbol{v}=' \
                + latex(self.S.cart*self.v) \
                + r'\right.\right\}' 
        else:
            return r'\left\{ \mathsf{\pmb{M}}\in\mathbb{R}^{' \
                + latex(self.S.Rn[0]) + r'\times' + latex(self.S.Rn[1]) + '}' \
                + r'\ \left|\ ' \
                + latex(self.S.cart) \
                + r'vec(\mathsf{\pmb{M}})=' \
                + latex(self.S.cart*self.v.vec()) \
                + r'\right.\right\}' 
        
    def latex(self):
        """ Construye el comando LaTeX para un EAfin de Rn"""
        return self.EcParametricas() + '\\; = \\;' + self.EcCartesianas()
            
    


class Homogenea:
    def __init__(self, sistema, rep=0, sust=[], repsust=0):
        """Resuelve un Sistema de Ecuaciones Lineales Homogéneo
        
        y muestra los pasos para encontrarlo"""
        try:
            A = Matrix(sistema).subs(sust)
        except:
            A = BlockM([sistema]).subs(sust)

        MA = A.apila(I(A.n),1)
        MA.corteElementos.update({sistema.n+A.m})

        K    = A.K(0, sust)  
        E    = I(A.n) & T(K.pasos[1])
        
        self.base        = Sistema([ v for j, v in enumerate(E,1) if (K|j).es_nulo(sust) ])
        self.sgen        = self.base if self.base else Sistema([ V0(sistema.n) ])
        self.determinado = (len(self.base) == 0)
        self.pasos       = K.pasos; 
        self.TrF         = K.TrF 
        self.TrC         = K.TrC

        self.enulo       = SubEspacio(self.sgen)
        
        if repsust:
            self.tex         = rprElim( A.apila( I(A.n) ,1 ) , self.pasos, [], sust)
        else:
            self.tex         = rprElim( A.apila( I(A.n) ,1 ) , self.pasos)
            
        if rep:
            display(Math(self.tex))
            
    
    def __repr__(self):
        """Muestra el Espacio Nulo de una matriz en su representación Python"""
        return 'Combinaciones lineales de (' + repr(self.sgen) + ')'
    
    def _repr_html_(self):
        """Construye la representación para el entorno Jupyter Notebook"""
        return html(self.latex())
    
    def latex(self):
        """ Construye el comando LaTeX para la solución de un Sistema Homogéneo"""
        if self.determinado:
            return '\\left\\{\ ' + latex(self.sgen|1) + '\ \\right\\}'
        else:
            return '\\mathcal{L}\\left(\ ' + latex(self.sgen) + '\ \\right)' 
       
    
class SEL:
    def __init__(self, sistema, b, rep=0, sust=[], repsust=0):
        """Resuelve un Sistema de Ecuaciones Lineales
        
        mediante eliminación con el sistema ampliado y muestra los pasos
        dados
        
        """
        try:
            A = Matrix(sistema.amplia(-b)).subs(sust).ccol({sistema.n})
        except:
            A = BlockM([sistema.amplia(-b)]).subs(sust).ccol({sistema.n})
            
        MA = A.apila(I(A.n),1)
        MA.corteElementos.update({sistema.n+A.m})
        operaciones = A.elim(0,False,sust).pasos[1]
        
        
        E         = I(sistema.n) & T(operaciones)
        
        testigo   = 0| (I(A.n) & T(operaciones)) |0
        Normaliza = T([]) if testigo==1 else T([( fracc(1,testigo), A.n )])
        pasos     = [[], operaciones+[Normaliza] ] if Normaliza else [[], operaciones]
        
        K         = A & T(operaciones)
        
        self.base        = Sistema([ v for j, v in enumerate(E,1) if (K|j).es_nulo(sust) ])
        self.sgen        = self.base if self.base else Sistema([ V0(sistema.n) ])
        self.determinado = (len(self.base) == 0)
        self.pasos       = pasos 
        self.TrF         = T(self.pasos[0]) 
        self.TrC         = T(self.pasos[1])
        
        if (K|0).no_es_nulo():
            self.solP  = set()
            self.eafin = set()
        else:
            self.solP  = (I(sistema.n).amplia(V0(sistema.n)) & T(pasos[1]))|0 
            self.eafin = EAfin(self.sgen, self.solP, 1)
        
        if repsust:
            self.tex     = rprElim( MA, self.pasos, [], sust )
        else:
            self.tex     = rprElim( MA, self.pasos)
        
        if rep:
            display(Math(self.tex))           
        
        
    
    def __repr__(self):
        """Muestra el Espacio Nulo de una matriz en su representación Python"""
        return repr(self.solP) + ' + Combinaciones lineales de (' + repr(self.sgen) + ')'
    
    def _repr_html_(self):
        """Construye la representación para el entorno Jupyter Notebook"""
        return html(self.latex())
    
    def latex(self):
        """ Construye el comando LaTeX para la solución de un Sistema Homogéneo"""
        if self.determinado and self.solP:
            return '\\left\\{\ ' + latex(self.solP) + '\ \\right\\}'
        else:
            return self.eafin.EcParametricas() if self.solP else '\\emptyset' #latex(set())
    
    

class Determinante:
   """Determinante de una Matrix mediante eliminación Gaussiana por columnas
   
   La representación muestra los pasos dados
   
   """
   def __init__(self, data, disp=0, sust=[]):
      
      def calculoDet(A, sust=[]):
          
          producto  = lambda x: 1 if not x else x[0] * producto(x[1:])
          
          def productos_realizados(operaciones):
              P = [ producto([-1 if isinstance(abv,set) else abv[0] \
                              for op in paso for abv in filter( lambda x: len(x)==2, op.abreviaturas)]) for paso in operaciones]
              return P
          
          operacionesEnColumnas = (A.L(0,sust).pasos[1])
          operacionesEnFilas    = [T((fracc(1,d),A.n+1)) for d in productos_realizados(operacionesEnColumnas)]
          pasos                 = [operacionesEnFilas, operacionesEnColumnas]
          
          matrixExtendida       = T(operacionesEnFilas) & A.extDiag(I(1),1) & T(operacionesEnColumnas)
          
          determinante          = sympy.sympify( producto( matrixExtendida.diag() ) ).simplify()
          
          tex                   = rprElimFyC( A.extDiag(I(1),1), pasos)
          
          return [tex, determinante, pasos]
          
      
      A  = Matrix(data.subs(sust))
      
      if not A.es_cuadrada():  raise ValueError('Matrix no cuadrada')
      
      self.tex, self.valor, self.pasos = calculoDet( A.subs(sust) )
      
      if disp:
         display(Math(self.tex))

   
   def __repr__(self):
      """Muestra un Sistema en su representación Python"""
      return 'Valor del determinante: ' + repr (self.valor)
   
   def _repr_html_(self):
      """Construye la representación para el entorno Jupyter Notebook"""
      return html(self.latex())
   
   def latex(self):
      """Construye el comando LaTeX para representar un Sistema"""
      return latex(self.valor)
   
         

class DiagonalizaS(Matrix):
    def __init__(self, sistema, espectro, rep=0, repType=0):
        """Diagonaliza por bloques triangulares una Matrix cuadrada
        
        Encuentra una matriz diagonal semejante mediante trasformaciones de
        sus columnas y las correspondientes transformaciones inversas espejo
        de las filas. Requiere una lista de autovalores (espectro), que deben
        aparecer en dicha lista tantas veces como sus respectivas
        multiplicidades algebraicas. Los autovalores aparecen en la diagonal
        principal de la matriz diagonal. El atributo S de dicha matriz
        diagonal es una matriz cuyas columnas son autovectores de los
        correspondientes autovalores.  """
        
        def texYpasos(data, pasos, rep=0, sust=[], repsust=0):
            pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
            TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
            if repsust:
                tex = rprElim(data, pasos, TexPasosPrev, sust)
            else:
                tex = rprElim(data, pasos, TexPasosPrev)
            pasos[0] = pasos[0] + pasosPrevios[0] 
            pasos[1] = pasosPrevios[1] + pasos[1]
            
            if rep:
                display(Math(tex))
            
            return tex, pasos
        
        A = Matrix(sistema)
        D = A.diagonalizaS(espectro)
        
        if rep:
            nan   = sympy.symbols('\ ')
            dummyMatrix = M1(A.n)*nan
            if repType==1:
                nan   = sympy.symbols('\ ')
                D.tex = rprElimCF(A.apila(I(A.n),1), D.pasos)
            elif repType==2:
                nan   = sympy.symbols('\ ')
                D.tex = rprElimCF(A.apila(I(A.n),1).concatena(I(A.m).apila(dummyMatrix,1),1), D.pasos)
            elif repType==3:
                nan   = sympy.symbols('\ ')
                D.tex = rprElimFyC(A.apila(I(A.n),1), D.pasos)
            elif repType==4:
                nan   = sympy.symbols('\ ')
                D.tex = rprElimFyC(A.apila(I(A.n),1).concatena(I(A.m).apila(dummyMatrix,1),1), D.pasos)
                
            display(Math(D.tex))
            
        self.__dict__.update(D.__dict__)
        self.__class__ = type(sistema)


class DiagonalizaO(Matrix):
    def __init__(self, sistema, espectro):
        """Diagonaliza ortogonalmente una Matrix simétrica
        
        Encuentra una matriz diagonal por semejanza empleando una matriz
        ortogonal Q a la derecha y su inversa (transpuesta) por la izquierda.
        Requiere una lista de autovalores (espectro), que deben aparecer
        tantas veces como sus respectivas multiplicidades algebraicas. Los
        autovalores aparecen en la diagonal principal de la matriz
        diagonal. El atributo Q de la matriz diagonal es la matriz ortogonal
        cuyas columnas son autovectores de los correspondientes autovalores.
        
        """
        A = Matrix(sistema)
        D = A.diagonalizaO(espectro)
            
        self.__dict__.update(D.__dict__)
        self.__class__ = type(sistema)


class DiagonalizaC(Matrix):
    def __init__(self, sistema, rep=False, sust=[], variante=0):
        """Diagonaliza por congruencia una Matrix simétrica (evitando dividir)
        
        Encuentra una matriz diagonal por conruencia empleando una matriz B
        invertible (evitando fracciones por defecto, si variante=1 entonces no
        evita las fracciones) por la derecha y su transpuesta por la
        izquierda. No emplea los autovalores. En general los elementos en la
        diagonal principal no son autovalores, pero hay tantos elementos
        positivos en la diagonal como autovalores positivos, tantos negativos
        como autovalores negativos, y tantos ceros como auntovalores nulos.
        
        """
        A = Matrix(sistema)
        D = A.diagonalizaC(rep, sust, variante)
            
        self.__dict__.update(D.__dict__)
        self.__class__ = type(sistema)
