# coding=utf8
import sympy
from IPython.display import display, Math
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
    try:
        return a.latex()
    except:
        return sympy.latex(simplifica(a))
         
def simplifica(self):
    """Devuelve las expresiones simplificadas"""
    if isinstance(self, (list, tuple, Sistema)):
        return type(self)([ simplifica(e) for e in self ])
    else:
        return (sympy.sympify(self)).simplify()
    
def filtradopasos(pasos):
    abv = pasos.t if isinstance(pasos,T) else pasos
           
    p = [T([j for j in T([abv[i]]).t if (isinstance(j,set) and len(j)>1)\
               or (isinstance(j,tuple) and len(j)==3 and j[0]!=0)       \
               or (isinstance(j,tuple) and len(j)==2 and j[0]!=1) ])    \
                                             for i in range(0,len(abv)) ]

    abv = [ t for t in p if t.t] # quitamos abreviaturas vacías de la lista
    
    return T(abv) if isinstance(pasos,T) else abv

def pinta(data):
    display(Math(latex(simplifica(data))))

           
class Sistema:
    """Clase Sistema

    Un Sistema es una lista ordenada de objetos. Los Sistemas se instancian
    con una lista, tupla u otro Sistema. 

    Parámetros:
        data (list, tuple, Sistema): lista, tupla o Sistema de objetos.

    Atributos:
        lista (list): lista de objetos.

    Ejemplos:
    >>> # Crea un nuevo Sistema a partir de una lista, tupla o Sistema

    >>> Sistema( [ 10, 'hola', T({1,2}) ]  )           # con lista
    >>> Sistema( ( 10, 'hola', T({1,2}) )  )           # con tupla
    >>> Sistema( Sistema( [ 10, 'hola', T({1,2}) ] ) ) # con Sistema

    [10; 'hola'; T({1, 2})]
    """
    def __init__(self, data):
        """Inicializa un Sistema con una lista, tupla o Sistema"""                        
        if isinstance(data, (list, tuple, Sistema)):
                            
            self.lista = list(data)
                            
        else:
            raise ValueError(' El argumento debe ser una lista, tupla, o Sistema.')

    def __getitem__(self,i):
        """ Devuelve el i-ésimo coeficiente del Sistema """
        return self.lista[i]

    def __setitem__(self,i,value):
        """ Modifica el i-ésimo coeficiente del Sistema """
        self.lista[i]=value
            
    def __len__(self):
        """Número de elementos del Sistema """
        return len(self.lista)

    def copy(self):
        """ Copia la lista de otro Sistema"""
        return type(self)(self.lista.copy())
            
    def __eq__(self, other):
        """Indica si es cierto que dos Sistemas son iguales"""
        return self.lista == other.lista

    def __ne__(self, other):
        """Indica si es cierto que dos Sistemas son distintos"""
        return self.lista != other.lista

    def __reversed__(self):
        """Devuelve el reverso de un Sistema"""
        return type(self)(list(reversed(self.lista)))
        
    def concatena(self,other,c=0):
        """ Concatena dos Sistemas """
        if not isinstance(other, Sistema):
            raise ValueError('Un Sistema solo se puede concatenar a otro Sistema')
        S = type(self)(self.lista + other.lista)
        if isinstance(other, Matrix) and c:
            S.cF, S.cC = self.cF, self.cC
            S.cC.update({self.n})
        return S

    def sis(self):
        return Sistema(self.lista)

    def __or__(self,j):
        """
        Extrae el j-ésimo componente del Sistema; o crea un Sistema con los
        elementos indicados (los índices comienzan por el número 1)

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

        [Vector([1, 0]), Vector([3, 0])] """
        if isinstance(j, int):
            return self[j-1]
            
        elif isinstance(j, (list,tuple) ):
            return type(self) ([ self|a for a in j ])

        elif isinstance(j, slice):
            start = None if j.start is None else j.start-1 
            stop  = None if j.stop  is None else (j.stop if j.stop>0 else j.stop-1)
            step  = j.step  or 1
            return type(self) (self[slice(start,stop,step)])

    def __add__(self, other):
        """Devuelve el Sistema resultante de sumar dos Sistemas

        Parámetros: 
            other (Sistema): Otro sistema del mismo tipo y misma longitud

        Ejemplos:
        >>> Sistema([10, 20, 30]) + Sistema([-1, 1, 1])

        Sistema([9, 21, 31]) 
        >>> Vector([10, 20, 30]) + Vector([-1, 1, 1])

        Vector([9, 21, 31]) 
        >>> Matrix([[1,5],[5,1]]) + Matrix([[1,0],[0,1]]) 

        Matrix([Vector([2, 5]); Vector([5, 2])]) """
        if not type(self)==type(other) or not len(self)==len(other):
            raise ValueError ('Solo se suman Sistemas del mismo tipo y misma longitud')

        return type(self) ([ (self|i) + (other|i) for i in range(1,len(self)+1) ])
                
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

        return type(self) ([ (self|i) - (other|i) for i in range(1,len(self)+1) ])
                
    def __neg__(self):
        """Devuelve el opuesto de un Sistema"""
        return -1*self

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
        if isinstance(x, (int, float, sympy.Basic)):
            return type(self)( [ x*(self|i) for i in range(1,len(self)+1) ] )
            
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
        if isinstance(x, (int, float, sympy.Basic)):
            return x*self

        elif isinstance(x, Vector):
            if len(self) != x.n:    raise ValueError('Sistema y Vector incompatibles')
            return sum([(self|j)*(x|j) for j in range(1,len(self)+1)], 0*self|1)

        elif isinstance(x, Matrix):
            if len(self) != x.m:      raise ValueError('Sistema y Matrix incompatibles')
            if isinstance(self, Vector):
                return Vector( [ self*(x|j) for j in range(1,(x.n)+1)], rpr='fila' )
            else:
                return type(self) ( [ self*(x|j) for j in range(1,(x.n)+1)] )

    def __and__(self,t):
        """Transforma los elementos de un Sistema S

        Atributos:
            t (T): transformaciones a aplicar sobre un Sistema S
        Ejemplos:
        >>>  S & T({1,3})                # Intercambia los elementos 1º y 3º
        >>>  S & T((5,1))                # Multiplica por 5 el primer elemento
        >>>  S & T((5,2,1))              # Suma 5 veces el elem. 1º al elem. 2º
        >>>  S & T([{1,3},(5,1),(5,2,1)])# Aplica la secuencia de transformac.
                     # sobre los elementos de S y en el orden de la lista
        """
        if isinstance(t.t,set):
            self.lista = [ (self|max(t.t)) if k==min(t.t) else \
                           (self|min(t.t)) if k==max(t.t) else \
                           (self|k)                 for k in range(1,len(self)+1)].copy()

        elif isinstance(t.t,tuple) and (len(t.t) == 2):
            self.lista = [ (t.t[0])*(self|k)                  if k==t.t[1] else \
                           (self|k)                 for k in range(1,len(self)+1)].copy()

        elif isinstance(t.t,tuple) and (len(t.t) == 3):
            self.lista = [ (t.t[0])*(self|t.t[1]) + (self|k)  if k==t.t[2] else \
                           (self|k)                 for k in range(1,len(self)+1)].copy()
        
        elif isinstance(t.t,list):
            for k in t.t:          
                self & T(k)
        
        return self
            
    def __repr__(self):
        """ Muestra un Sistema en su representación python """
        pc = ';' if len(self.lista) else ''
        return 'Sistema([' + \
            '; '.join( repr (e) for e in self ) + \
            pc + '])'

    def _repr_html_(self):
        """ Construye la representación para el entorno jupyter notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para representar un Sistema """
        pc = ';' if len(self.lista) else '\\ '
        return '\\left[' + \
            ';\;'.join( latex(e) for e in self ) + \
            pc + '\\right]'

    def de_composicion_uniforme(self):
       """Indica si es cierto que todos los elementos son del mismo tipo"""
       return all(type(e)==type(self[0]) for e in self)
    def es_nulo(self):
        """Indica si es cierto que el Sistema es nulo"""
        return self==self*0

    def no_es_nulo(self):
        """Indica si es cierto que el Sistema no es nulo"""
        return self!=self*0

    def junta(self, l):
        """Junta una lista o tupla de Sistemas en uno solo concatenando las
        correspondientes listas de los distintos Sistemas"""
        l = l if isinstance(l, list) else [l]

        junta_dos = lambda x, other: x.concatena(other)
        reune     = lambda x: x[0] if len(x)==1 else junta_dos( reune(x[0:-1]), x[-1] )
        
        return reune([self] + [s for s in l])
        
    def subs(self, s,v):
        if isinstance(self, sympy.Basic):
            return sympy.S(self).subs(s,v)
        elif isinstance(self, Sistema):
            return type(self)([ sympy.S(e).subs(s,v) for e in self ])
        
class Vector(Sistema):
    """Clase Vector(Sistema)

    Vector es un Sistema de números u objetos de la librería Sympy. Se puede
    instanciar con una lista, tupla o Sistema. Si se instancia con un Vector
    se crea una copia del mismo. El atributo 'rpr' indica al entorno Jupyter 
    si el vector debe ser escrito como fila o como columna.

    Parámetros:
        sis (list, tuple, Sistema, Vector) : Lista, tupla o Sistema de 
            objetos de tipo int, float o sympy.Basic, o bien otro Vector.
        rpr (str) : Representación en Jupyter ('columna' por defecto).
            Si rpr='fila' el vector se representa en forma de fila. 

    Atributos:
        n     (int)    : número de elementos de la lista.
        rpr   (str)    : modo de representación en Jupyter.

    Atributos heredados de la clase Sistema:
        lista (list)   : list con los elementos.

    Ejemplos:
    >>> # Instanciación a partir de una lista, tupla o Sistema de números
    >>> Vector( [1,2,3] )           # con lista
    >>> Vector( (1,2,3) )           # con tupla
    >>> Vector( Sistema( [1,2,3] ) )# con Sistema
    >>> Vector( Vector ( [1,2,3] ) )# a partir de otro Vector

    Vector([1,2,3])
    """        
    def __init__(self, data, rpr='columna'):
        """Inicializa Vector con una lista, tupla o Sistema"""
                            
        super().__init__(data)
                            
        if not all( [isinstance(e, (int, float, sympy.Basic)) for e in self] ):
            raise ValueError('no todos los elementos son números o parámetros!')
            
                            
        self.rpr  =  rpr    
        self.n    =  len(self)

    def __ror__(self,i):
        """Hace exactamente lo mismo que el método __or__ por la derecha."""
        return self | i
        
    def __rand__(self,t):
        """Hace exactamente lo mismo que el método __and__ por la derecha."""
        return self & t
        
    def diag(self):
        """Crea una Matrix diagonal cuya diagonal es self"""
        return Matrix([a*(I(self.n)|j) for j,a in enumerate(self, 1)])

    def norma(self):
        """Devuelve un múltiplo de un vector (no nulo) pero norma uno"""
        return sympy.sqrt(self*self)
                                                                   
    def normalizado(self):
        """Devuelve un múltiplo de un vector (no nulo) pero norma uno"""
        if self.es_nulo(): raise ValueError('Un vector nulo no se puede normalizar')

        return self * fracc(1,self.norma())
            
    def __repr__(self):
        """ Muestra el vector en su representación Python """
        return 'Vector(' + repr(self.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el entorno Jupyter Notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para representar un Vector"""
        if self.rpr == 'fila' or self.n==1:    
            return '\\begin{pmatrix}' + \
                   ',&'.join([latex(e) for e in self]) + \
                   ',\\end{pmatrix}' 
        else:
            return '\\begin{pmatrix}' + \
                   '\\\\'.join([latex(e) for e in self]) + \
                   '\\end{pmatrix}'
                   
    
class Matrix(Sistema):
    """Clase Matrix

    Es un Sistema de Vectores con el mismo número de componentes. Una Matrix
    se puede construir con: 1) una lista, tupla o Sistema de Vectores con el
    mismo número de componentes (serán las columnas); 2) una lista, tupla o 
    Sistema de listas, tuplas o Sistemas con el mismo número de componentes 
    (serán las filas de la matriz); 3) una Matrix (se obtendrá una copia); 
    4) una BlockM (se obtendrá la Matrix que resulta de unir los bloques).

    Parámetros:
        data (list, tuple, Sistema, Matrix, BlockM): Lista, tupla o Sistema 
        de Vectores (columnas con mismo núm. de componentes); o de listas,
        tuplas o Sistemas (filas de misma longitud); o una Matrix o BlockM.

    Atributos:
        m     (int)    : número de filas de la matriz
        n     (int)    : número de columnas de la matriz

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
    >>> # Crea una Matrix a partir de una BlockM
    >>> Matrix( {1}|A|{2} )

    Matrix([ Vector([1, 2]); Vector([1, 0]); Vector([9, 2]) ]) """
    def __init__(self, data):
        """Inicializa una Matrix"""
        super().__init__(data)
        
        lista = Sistema(data).lista
        
        if isinstance(lista[0], Vector):
            if not all ( isinstance(v, Vector)   and lista[0].n==v.n       for v in lista ):
                raise ValueError('no todo son vectores, o no tienen la misma longitud!')

            self.lista   = lista.copy()
            
        elif isinstance(data, SisMat):
            self.lista = (data|1).apila(data.lista[1:]).lista.copy() if len(data)>1 \
                                            else (data|1).lista.copy()

        elif isinstance(data, BlockM):
            listmat = [Matrix(sismat) for sismat in data]
            self.lista = (listmat[0]).junta(listmat[1:]).lista.copy()

        elif isinstance(lista[0], (list, tuple, Sistema)):
            if not all ( type(lista[0])==type(v) and len(lista[0])==len(v) for v in lista ):
                raise ValueError('no todo son listas, o no tienen la misma longitud!')

            self.lista  =  [ Vector( [ lista[i][j] for i in range(len(lista)) ] ) \
                                                   for j in range(len(lista[0])) ].copy()

        
        self.m  = len(self|1)
        self.n  = len(self)
        self.cC = {0}
        self.cF = {0}
                            
    def __or__(self,j):
        """
        Extrae la i-ésima columna de Matrix; o crea una Matrix con las columnas
        indicadas; o crea una BlockM particionando una Matrix por las
        columnas indicadas (los índices comienzan por la posición 1)

        Parámetros:
            j (int, list, tuple, slice): Índice (o lista de índices) de la 
                  columna (o columnas) a seleccionar
              (set): Conjunto de índices de las columnas por donde particionar

        Resultado:
            Vector: Cuando j es int, devuelve la columna j-ésima de Matrix.
            Matrix: Cuando j es list, tuple o slice, devuelve la Matrix formada 
                por las columnas indicadas en la lista o tupla de índices.
            BlockM: Si j es un set, devuelve la BlockM resultante de particionar
                la matriz a la derecha de las columnas indicadas en el conjunto

        Ejemplos:
        >>> # Extrae la j-ésima columna la matriz 
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | 2

        Vector([0, 2])
        >>> # Matrix formada por Vectores columna indicados en la lista (o tupla)
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | [2,1]
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | (2,1)

        Matrix( [Vector([0, 2]); Vector([1, 0])] )
        >>> # BlockM correspondiente a la partición por la segunda columna
        >>> Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) | {2}

        BlockM([SisMat([Matrix([Vector([1, 0]), Vector([0, 2])])]), 
                SisMat([Matrix([Vector([3, 0])])])])
        """
        if isinstance(j, int):
            return self[j-1]
            
        elif isinstance(j, (list,tuple) ):
            return type(self) ([ self|a for a in j ])

        elif isinstance(j, slice):
            start = None if j.start is None else j.start-1 
            stop  = None if j.stop  is None else (j.stop if j.stop>0 else j.stop-1)
            step  = j.step  or 1
            return type(self) (self[slice(start,stop,step)])
        elif isinstance(j,set):
            return BlockM ([ [self|a for a in particion(j,self.n)] ])
             
    def __invert__(self):
        """
        Devuelve la traspuesta de una matriz.

        Ejemplo:
        >>> ~Matrix([ [1,2,3] ])

        Matrix([ Vector([1, 2, 3]) ])
        """
        M = Matrix ([ c.lista for c in self ])
        M.cF, M.cC = self.cC, self.cF
        return M
        
    def __ror__(self,i):
        """Operador selector por la izquierda

        Extrae la i-ésima fila de Matrix; o crea una Matrix con las filas 
        indicadas; o crea una BlockM particionando una Matrix por las filas
        indicadas (los índices comienzan por la posición 1)

        Parámetros:
            i (int, list, tuple): Índice (o índices) de las filas a seleccionar
              (set): Conjunto de índices de las filas por donde particionar

        Resultado:
            Vector: Cuando i es int, devuelve la fila i-ésima de Matrix.
            Matrix: Cuando i es list o tuple, devuelve la Matrix cuyas filas son
                las indicadas en la lista de índices.
            BlockM: Cuando i es un set, particiona la matriz por debajo de las 
                filas indicadas en el conjunto.

        Ejemplos:
        >>> # Extrae la j-ésima fila de la matriz 
        >>> 2 | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])])

        Vector([0, 2, 0])
        >>> # Matrix formada por Vectores fila indicados en la lista (o tupla)
        >>> [1,1] | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])]) 
        >>> (1,1) | Matrix([Vector([1,0]), Vector([0,2]), Vector([3,0])])

        Matrix([Vector([1, 1]), Vector([0, 0]), Vector([3, 3])])
        >>> # BlockM correspondiente a la partición por la primera fila
        >>> {1} | Matrix([Vector([1,0]), Vector([0,2])])

        BlockM([ SisMat([Matrix([Vector([1]), Vector([0])]), 
                         Matrix([Vector([0]), Vector([2])])]) ])
        """
        if isinstance(i,int):
            return  Vector( (~self)|i , rpr='fila' )

        elif isinstance(i, (list,tuple,slice)):        
            return ~Matrix( (~self)|i ) 
        
        elif isinstance(i,set):
            return BlockM ([ [a|self] for a in particion(i,self.m) ])
            

    def __rand__(self,t):
        """Transforma las filas de una Matrix

        Atributos:
            t (T): transformaciones a aplicar sobre las filas de Matrix

        Ejemplos:
        >>>  T({1,3})   & A               # Intercambia las filas 1 y 3
        >>>  T((5,1))   & A               # Multiplica por 5 la fila 1
        >>>  T((5,2,1)) & A               # Suma 5 veces la fila 2 a la fila 1
        >>>  T([(5,2,1),(5,1),{1,3}]) & A # Aplica la secuencia de transformac.
                    # sobre las filas de A y en el orden inverso al de la lista
        """
        
        if isinstance(t.t, (set, tuple) ):
            self.lista = (~(~self & t)).lista.copy()
        
        elif isinstance(t.t,list):
            for k in reversed(t.t):          
                T(k) & self
        
        return self 
        
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
        """Crea un Vector a partir de la diagonal de self"""
        return Vector([ (j|self|j) for j in range(1,min(self.m,self.n)+1)])

    def normalizada(self,o='Columnas'):
        if o == 'Columnas':
            if any( v.es_nulo() for v in self):
                raise ValueError('algún vector es nulo')
            return Matrix([ v.normalizado() for v in self])
        else:
            return ~(~self.normalizada())
            
    def apila(self, l, c=0):
        """Apila una lista o tupla de Matrix con el mismo número de columnas
        en una única Matrix concatenando las respectivas columnas"""
        l = l if isinstance(l, list) else [l]
        apila_dos = lambda x, other, c=0: ~((~x).concatena(~other,c))
        apila = lambda x: x[0] if len(x)==1 else apila_dos( apila(x[0:-1]), x[-1] , c )
        
        return apila([self] + [s for s in l])
        
    def ExtiendeDiag(self,lista):
        if not all(isinstance(m,Matrix) for m in lista): 
            return ValueError('No es una lista de matrices')
        Ext_dos = lambda x, y: Matrix(BlockM([[x,M0(x.m,y.n)],[M0(y.m,x.n),y] ]))
        ExtDiag     = lambda x: x[0] if len(x)==1 else Ext_dos( ExtDiag(x[0:-1]), x[-1] )
        return ExtDiag([self]+lista)
        
    def extDiag(self,lista,c=0):
        def CreaLista(t):
            """Devuelve t si t es una lista; si no devuelve la lista [t]"""
            return t if isinstance(t, list) else [t]
            
        lista = CreaLista(lista)
        if not all(isinstance(m,Matrix) for m in lista): 
            return ValueError('No es una lista de matrices')
        Ext_dos = lambda x, y: x.apila(M0(y.m,x.n),c).concatena(M0(x.m,y.n).apila(y,c),c)
        ExtDiag     = lambda x: x[0] if len(x)==1 else Ext_dos( ExtDiag(x[0:-1]), x[-1] )
        return ExtDiag([self]+lista)
        
    def BlockDiag(self,lista):
        if not all(isinstance(m,Matrix) for m in lista): 
            return ValueError('No es una lista de matrices')
        
        lm = [e.m for e in [self]+lista]
        ln = [e.n for e in [self]+lista]
        return key(lm)|self.ExtiendeDiag(lista)|key(ln)
        
    def es_singular(self):
        if not self.es_cuadrada():
            raise ValueError('La matriz no es cuadrada')
        return self.rg()<self.n
      
    def K(self,rep=0):
        """Una forma pre-escalonada por columnas (K) de una Matrix"""
        return Elim(self,rep)
        
    def L(self,rep=0): 
        """Una forma escalonada por columnas (L) de una Matrix"""
        return ElimG(self,rep)
        
    def U(self,rep=0): 
        """Una forma escalonada por columnas (L) de una Matrix"""
        return ElimGF(self,rep)

    def R(self,rep=0):
        """Forma escalonada reducida por columnas (R) de una Matrix"""
        return ElimGJ(self,rep)

    def rg(self):
        """Rango de una Matrix"""
        return self.K().rango

    def determinante(self, rep=0):
        """Devuelve el valor del det. de Matrix"""
        return Determinante(self,rep).valor

    def inversa(self, rep=0):                                                               
        """Inversa de Matrix"""
        return InvMat(self,rep)

    def diagonalizaS(self, espectro, rep=0):
        """Diagonaliza por bloques triangulares una Matrix cuadrada 

        Encuentra una matriz diagonal semejante mediante trasformaciones de sus
        columnas y las correspondientes transformaciones inversas espejo de las
        filas. Requiere una lista de autovalores (espectro), que deben aparecer
        en dicha lista tantas veces como sus respectivas multiplicidades 
        algebraicas. Los autovalores aparecen en la diagonal principal de la 
        matriz diagonal. El atributo S de dicha matriz diagonal es una matriz 
        cuyas columnas son autovectores de los correspondientes autovalores.
        """
        return DiagonalizaS(self, espectro, rep)

    def diagonalizaO(self, espectro, rep=0):
        """ Diagonaliza ortogonalmente una Matrix simétrica 

        Encuentra una matriz diagonal por semejanza empleando una matriz
        ortogonal Q a la derecha y su inversa (transpuesta) por la izquierda. 
        Requiere una lista de autovalores (espectro), que deben aparecer tantas
        veces como sus respectivas multiplicidades algebraicas. Los autovalores 
        aparecen en la diagonal principal de la matriz diagonal. El atributo Q 
        de la matriz diagonal es la matriz ortogonal cuyas columnas son 
        autovectores de los correspondientes autovalores. """
        return DiagonalizaO(self, espectro)

    def diagonalizaC(self, rep=0):
        """ Diagonaliza por congruencia una Matrix simétrica (evitando dividir)

        Encuentra una matriz diagonal por conruencia empleando una matriz B 
        invertible (y entera si es posible) por la derecha y su transpuesta por
        la izquierda. No emplea los autovalores. En general los elementos en la
        diagonal principal no son autovalores, pero hay tantos elementos 
        positivos en la diagonal como autovalores positivos, tantos negativos 
        como autovalores negativos, y tantos ceros como auntovalores nulos. """
        return DiagonalizaC(self, rep)

    def diagonalizaCr(self, rep=0):
        """ Diagonaliza por congruencia una Matrix simétrica

        Encuentra una matriz diagonal congruente multiplicando por una matriz
        invertible B a la derecha y por la transpuesta de B por la izquierda. 
        No requiere conocer los autovalores. En general los elementos en la
        diagonal principal de la matriz diagonal no son autovalores, pero hay
        tantos elementos positivos en la diagonal como autovalores positivos
        (incluyendo la multiplicidad de cada uno), tantos negativos como
        autovalores negativos (incluyendo la multiplicidad de cada uno), y tantos
        ceros como la multiplicidad algebraica del autovalor cero. """
        return DiagonalizaCr(self, rep)

    def __pow__(self,n):
        """Calcula la n-ésima potencia de una Matrix"""
        if not isinstance(n,int): raise ValueError('La potencia no es un entero')
        if not self.es_cuadrada:  raise ValueError('Matrix no es cuadrada')

        M = self if n else I(self.n)
        for i in range(1,abs(n)):
            M = M * self

        return M.inversa() if n < 0 else M

    def det(self):
        """Calculo del determinate mediante la expansión de Laplace"""
        if not self.es_cuadrada(): raise ValueError('Matrix no cuadrada')
                                                                   
        def cof(self,f,c):
            """Cofactor de la fila f y columna c"""
            excl = lambda k: tuple(i for i in range(1,self.m+1) if i!=k)
            return (-1)**(f+c)*(excl(f)|self|excl(c)).det()
                                                                   
        if self.m == 1:
            return 1|self|1
            
        return sum([(f|self|1)*cof(self,f,1) for f in range(1,self.m+1)]) # columna 1
                                                                                                                                  
    def GS(self):
        """Devuelve una Matrix equivalente cuyas columnas son ortogonales

        Emplea el método de Gram-Schmidt"""
        A = Matrix(self)
        for n in range(2,A.n+1):
            A & T([ (-fracc((A|n)*(A|j),(A|j)*(A|j)), j, n) \
                    for j in range(1,n) if (A|j).no_es_nulo() ])
        return A

    def __repr__(self):
        """ Muestra una matriz en su representación Python """
        return 'Matrix(' + repr(self.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el  entorno Jupyter Notebook """
        return html(self.latex())
        
    def cfil(self,conjuntoIndices):
        """ Añade el atributo cfilas para insertar lineas horizontales """
        self.cF = set(conjuntoIndices) if conjuntoIndices else {0}
        return self

    def ccol(self,conjuntoIndices):
        """ Añade el atributo cfilas para insertar lineas horizontales """
        self.cC = set(conjuntoIndices) if conjuntoIndices else {0}
        return self

    def latex(self):
        """ Construye el comando LaTeX para representar una Matrix """
        ln = [len(n) for n in particion(self.cC,self.n)]                                                           
        return \
         '\\left[ \\begin{array}{' + '|'.join([n*'c' for n in ln])  + '}' + \
         '\\\\ \\hline '.join(['\\\\'.join(['&'.join([latex(e) for e in f.lista]) \
           for f in (~M).lista]) \
           for M in [ i|self for i in particion(self.cF,self.m)]]) + \
         '\\\\ \\end{array} \\right]'
        
    
class T:
    """Clase T

    T ("Transformación elemental") guarda en su atributo 't' una abreviatura
    (o una secuencia de abreviaturas) de transformaciones elementales. El 
    método __and__ actúa sobre otra T para crear una T que es composición de 
    transformaciones elementales (una la lista de abreviaturas), o bien actúa 
    sobre una Matrix (para transformar sus filas).

    Atributos:
        t (set)  : {índice, índice}. Abrev. de un intercambio entre los 
                     vectores correspondientes a dichos índices
          (tuple): (escalar, índice). Abrev. transf. Tipo II que multiplica
                     el vector correspondiente al índice por el escalar
                 : (escalar, índice1, índice2). Abrev. transformación Tipo I
                     que suma al vector correspondiente al índice2 el vector
                     correspondiente al índice1 multiplicado por el escalar
          (list) : Lista de conjuntos y tuplas. Secuencia de abrev. de
                     transformaciones como las anteriores. 
          (T)    : Transformación elemental. Genera una T cuyo atributo t es
                     una copia del atributo t de la transformación dada 
          (list) : Lista de transformaciones elementales. Genera una T cuyo 
                     atributo es la concatenación de todas las abreviaturas
    Ejemplos:
    >>> # Intercambio entre vectores
    >>> T( {1,2} )

    >>> # Trasformación Tipo II (multiplica por 5 el segundo vector)
    >>> T( (5,2) )

    >>> # Trasformación Tipo I (resta el tercer vector al primero)
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
    def __init__(self, t, rpr='v'):
        """Inicializa una transformación elemental"""
        def CreaLista(t):
            """Devuelve t si t es una lista; si no devuelve la lista [t]"""
            return t if isinstance(t, list) else [t]
            
        if isinstance(t, T):
            self.t = t.t

        elif isinstance(t, list) and t and isinstance(t[0], T): 
                self.t = [val for sublist in [x.t for x in t] for val in CreaLista(sublist)]

        else:
            self.t = t
        for j in CreaLista(self.t):
            if isinstance(j,tuple) and (len(j) == 2) and j[0]==0:
                raise ValueError('T( (0, i) ) no es una trasformación elemental')
            if isinstance(j,tuple) and (len(j) == 3) and (j[1] == j[2]):
                raise ValueError('T( (a, i, i) ) no es una trasformación elemental')
            if isinstance(j,set) and (len(j) > 2) or not j:
                raise ValueError \
                ('El conjunto debe tener uno o dos índices para ser un intercambio')
        self.rpr = rpr

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
        def CreaLista(t):
            """Devuelve t si t es una lista; si no devuelve la lista [t]"""
            return t if isinstance(t, list) else [t]
            
        if isinstance(other, T):
            return T(CreaLista(self.t) + CreaLista(other.t), self.rpr)

        if isinstance(other, (Vector, Matrix)):
            return other.__rand__(self)

    def __invert__(self):
        """Transpone la lista de abreviaturas (invierte su orden)"""
        return T( list(reversed(self.t)), self.rpr) if isinstance(self.t, list) else self
        
    def __pow__(self,n):
        """Calcula potencias de una T (incluida la inversa)"""
        def Tinversa ( self ):
            """Calculo de la inversa de una transformación elemental"""
            def CreaLista(t):
                """Devuelve t si t es una lista; si no devuelve la lista [t]"""
                return t if isinstance(t, list) else [t]
                

            listaT = [          j            if isinstance(j,set) else \
                       ( -j[0], j[1],  j[2]) if len(j)==3         else \
                       (fracc(1,j[0]), j[1])              for j in CreaLista(self.t) ]

            return ~T( listaT, self.rpr)
    
        if not isinstance(n,int):
            raise ValueError('La potencia no es un entero')

        potencia = lambda x, n: x if n==1 else x & potencia(x, n-1)
        t = potencia(self,abs(n)) if n!=0  else  T({1})
        
        return t if n>0 else Tinversa(t)
            
    def espejo ( self ):
        """Calculo de la transformación elemental espejo de otra"""
        def CreaLista(t):
            """Devuelve t si t es una lista; si no devuelve la lista [t]"""
            return t if isinstance(t, list) else [t]
            
        return T([(j[0],j[2],j[1]) if len(j)==3 else j for j in CreaLista(self.t)],self.rpr)
        
    def subs(self,c,v):
        self.t=[sympy.S(item).subs(c,v) for item in self.t]
        return self
    def __eq__(self, other):
        """Indica si es cierto que dos Transformaciones elementales son iguales"""
        return self.t == other.t
    def __repr__(self):
        """ Muestra T en su representación Python """
        return 'T(' + repr(self.t) + ')'

    def _repr_html_(self):
        """ Construye la representación para el entorno Jupyter Notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para representar una Trans. Elem. """
        def simbolo(t):
            """Escribe el símbolo que denota una trasformación elemental particular"""
            if isinstance(t,set):
                return '\\left[\\mathbf{' + latex(min(t)) + \
                  '}\\rightleftharpoons\\mathbf{' + latex(max(t)) + '}\\right]'
            if isinstance(t,(tuple, sympy.core.containers.Tuple)) and len(t) == 2:
                return '\\left[\\left(' + \
                  latex(t[0]) + '\\right)\\mathbf{'+ latex(t[1]) + '}\\right]'
            if isinstance(t,(tuple, sympy.core.containers.Tuple)) and len(t) == 3:
                return '\\left[\\left(' + latex(t[0]) + '\\right)\\mathbf{' + \
                  latex(t[1]) + '}' + '+\\mathbf{' + latex(t[2]) + '} \\right]'    

        if isinstance(self.t, (set, tuple) ):
            return '\\underset{' + simbolo(self.t) + '}{\\pmb{\\tau}}'

        elif self.t == []:
            return ' '

        elif isinstance(self.t, list) and self.rpr=='v':
            return '\\underset{\\begin{subarray}{c} ' + \
                  '\\\\'.join([simbolo(i) for i in self.t])  + \
                  '\\end{subarray}}{\\pmb{\\tau}}'

        elif isinstance(self.t, list):
            return '\\underset{' + \
                   '}{\\pmb{\\tau}}\\underset{'.join([simbolo(i) for i in self.t]) + \
                   '}{\\pmb{\\tau}}'
                  
                       
class V0(Vector):
    def __init__(self, n ,rpr = 'columna'):
        """ Inicializa el vector nulo de n componentes"""
        super().__init__([0 for i in range(n)], rpr)
        self.__class__ = Vector

class V1(Vector):
    def __init__(self, n ,rpr = 'columna'):
        """ Inicializa el vector nulo de n componentes"""
        super().__init__([1 for i in range(n)], rpr)
        self.__class__ = Vector

class M0(Matrix):
    def __init__(self, m, n=None):
        """ Inicializa una matriz nula de orden n """
        n = m if n is None else n

        super().__init__([ V0(m) for j in range(n)])
        self.__class__ = Matrix

class M1(Matrix):
    def __init__(self, m, n=None):
        """ Inicializa una matriz nula de orden n """
        n = m if n is None else n

        super().__init__([ V1(m) for j in range(n)])
        self.__class__ = Matrix

class I(Matrix):
    def __init__(self, n):
        """ Inicializa la matriz identidad de tamaño n """
        super().__init__([[(i==j)*1 for i in range(n)] for j in range(n)])
        self.__class__ = Matrix


def particion(s,n):
    """ genera la lista de particionamiento a partir de un conjunto y un número
    >>> particion({1,3,5},7)

    [[1], [2, 3], [4, 5], [6, 7]]
    """
    p = sorted(list(s | set([0,n])))
    return [ list(range(p[k]+1,p[k+1]+1)) for k in range(len(p)-1) ]
    
def key(L):
    """Genera el conjunto clave a partir de una secuencia de tamaños
    número
    >>> key([1,2,1])

    {1, 3, 4}
    """
    return set([ sum(L[0:i]) for i in range(1,len(L)+1) ])   

class SisMat(Sistema):
    def __init__(self, data):
        """Inicializa un SisMat con una lista, tupla o Sistema de Matrix,
        (todas con el mismo número de columnas).
        """
        super().__init__(data)

        lista = Sistema(data).lista.copy()
               
        if isinstance(data[0], Matrix):
            
            if not all( isinstance(m,Matrix) and m.n==lista[0].n for m in self):
                raise ValueError('O no todo son Matrix o no tienen el mismo número de columnas!')
            self.lista = lista.copy()
                                                                                
        self.n     = len(self)
               
        self.ln    = (self|1).n
        self.lm    = [matriz.m for matriz in self]

    def __or__(self, j):
        if isinstance(j, int):
            return self[j-1]
            
        elif isinstance(j, (list,tuple) ):
            return type(self) ([ self|a for a in j ])

        elif isinstance(j, slice):
            start = None if j.start is None else j.start-1 
            stop  = None if j.stop  is None else (j.stop if j.stop>0 else j.stop-1)
            step  = j.step  or 1
            return type(self) (self[slice(start,stop,step)])        
        elif isinstance(j, set):
            return BlockM([ SisMat( [Mat|list(c) for Mat in self] ) \
                                        for c in particion(j, self.ln) ])

    def __ror__(self,i):
        """Hace exactamente lo mismo que el método __or__ por la derecha 
        cuando es el argumento es int, list, tuple o slice. Cuando el 
        argumento es un conjunto se reparticiona por las filas indicadas por
        el conjunto"""
        if isinstance(i, (int, list, tuple, slice)):
            return self | i
        elif isinstance(i, set):
            return SisMat([ list(f)|Matrix(self) for f in particion(i, Matrix(self).m) ])
        
    def __repr__(self):
        """ Muestra un SisMat en su representación Python """
        return 'SisMat(' + repr(self.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el entorno Jupyter Notebook """
        return html(self.latex())

    def latex(self):
        """ Escribe el código de LaTeX para representar una SisMat """
        if self.n == 1:       
            return '\\begin{pmatrix}' + latex(self|1) + '\\end{pmatrix}'
            
        else:
            return \
              '\\left(\\!\\!\\!\\!\\left(' + \
              '\\begin{array}{' + self.ln*'c' + '}' + \
              '\\\\ \\hline\\hline'.join( ['\\\\'.join( ['&'.join( [latex(e) \
                     for e in fila ]) for fila in ~Mat ]) for Mat in self ]) + \
              '\\\\' + \
              '\\end{array}' + \
              '\\right)\\!\\!\\!\\!\\right)'

    
class BlockM(Sistema):
    def __init__(self, data):
        """Inicializa una BlockM con una lista, tupla, o Sistema: de SisMats 
        (serán las columnas de matrices) o bien de listas o tuplas de 
        matrices (filas de matrices)
        """
        super().__init__(data)

        lista = Sistema(data).lista
               
        if isinstance(lista[0], Sistema): 
            if not all ( isinstance(s, Sistema) and s.de_composicion_uniforme() and \
                         isinstance(s|1, Matrix) and  len(s)==len(lista[0]) for s in lista ):
                raise ValueError('no son Sistemas de matrices, o no tienen la misma longitud!')

            self.lista = [ SisMat(e) for e in lista ].copy()
                                                                     
        elif isinstance(data[0], (list, tuple)):
            if not all ( isinstance(s, (list,tuple)) and  isinstance(s[0], Matrix) and \
                         all(type(e)==type(lista[0]) for e in lista) and \
                         len(s)==len(lista[0]) for s in lista ):
                raise ValueError('no son listas de matrices, o no tienen la misma longitud!')

            self.lista  =  [ SisMat([ lista[j][i] for j in range(len(lista))]) \
                                                  for i in range(len(lista[0])) ].copy()
               
        self.m     = len(self|1)
        self.n     = len(self)
               
        self.lm    = (self|1).lm
        self.ln    = [sm.ln for sm in self]

    def __or__(self, j):
        if isinstance(j, int):
            return self[j-1]
            
        elif isinstance(j, (list,tuple) ):
            return type(self) ([ self|a for a in j ])

        elif isinstance(j, slice):
            start = None if j.start is None else j.start-1 
            stop  = None if j.stop  is None else (j.stop if j.stop>0 else j.stop-1)
            step  = j.step  or 1
            return type(self) (self[slice(start,stop,step)])        
        elif isinstance(j, set):
            if self.n == 1:
                return  (self|1)|j 
                                    
            elif self.n > 1: 
                 return (key(self.lm) | Matrix(self)) | j


    def __ror__(self,i):
        if isinstance(i, (int)):
            return BlockM( [ [i|sm for sm in self] ] )
        
        if isinstance(i, (list,tuple,slice,set) ):        
            return BlockM( [i|sm for sm in self] )
        
    def __repr__(self):
        """ Muestra una BlockM en su representación Python """
        return 'BlockM(' + repr(self.lista) + ')'

    def _repr_html_(self):
        """ Construye la representación para el  entorno Jupyter Notebook """
        return html(self.latex())

    def latex(self):
        """ Escribe el código de LaTeX para representar una BlockM """                     
        Neg  = '\!\!\!\!' if Matrix(self).m > 1 else '\!'
        Neg2 = '\\!' if len(self.ln) > 1 and Matrix(self).m == 2 else ''
        Neg3 = '\\!' if Matrix(self).m > 2 else ''
        Pos = '\,' if Matrix(self).m > 2 else ''
        return \
              '\,\,\,\\left[' + Neg + Neg2 + Neg3 + '\\left[' + Pos + \
              '\\begin{array}{' + '|'.join([n*'c' for n in self.ln])  + '}' + \
              '\\\\ \\hline '.join( ['\\\\'.join( ['&'.join( \
               [latex(e) for e in fila]) for fila in ~Mat]) for Mat in (self|{0}|1)]) + \
              '\\\\' + \
              '\\end{array}' + Pos +\
              '\\right]' + Neg + Neg2 + Neg3 + '\\right]\,\,\,'

    

class Elim(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma pre-escalonada de Matrix(data)

           operando con las columnas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        celim = lambda x: x > p
        A = Matrix(data);  r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T( [ T( [ ( denom((i|A|j),(i|A|p)),    j),    \
                               (-numer((i|A|j),(i|A|p)), p, j)  ] ) \
                                              for j in filter(celim, range(1,A.n+1)) ] )
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(p)
        pasos = [[], transformaciones]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix
        
class ElimG(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada de Matrix(data)

           operando con las columnas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        A = Elim(data);  r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ {p, r} ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(r)
        pasos = [ [], A.pasos[1]+[T(transformaciones)] ]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix

class ElimGJ(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada reducida de Matrix(data)

           operando con las columnas (y evitando operar con fracciones  
           hasta el último momento). Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        celim = lambda x: x < p
        A = ElimG(data);
        r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T( [ T( [ ( denom((i|A|j),(i|A|p)),    j),    \
                               (-numer((i|A|j),(i|A|p)), p, j)  ] ) \
                                              for j in filter(celim, range(1,A.n+1)) ] )
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(p)
                
        transElimIzda = transformaciones

        r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ (fracc(1, i|A|p), p) ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(p)
                
        pasos = [ [], A.pasos[1] + transElimIzda  + [T(transformaciones)] ]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix

class Elimr(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma pre-escalonada de Matrix(data)

           operando con las columnas. Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        celim = lambda x: x > p
        A = Matrix(data);  r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ (-fracc(i|A|j, i|A|p), p, j) for j in filter(celim, range(1,A.n+1)) ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(p)
        pasos = [[], transformaciones]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix
        
class ElimrG(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada de Matrix(data)

           operando con las columnas. Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        A = Elimr(data);  r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ {p, r} ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(r)
        pasos = [ [], A.pasos[1]+[T(transformaciones)] ]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix

class ElimrGJ(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada reducida de Matrix(data)

           operando con las columnas. Si rep es no nulo, se muestran en
           Jupyter los pasos dados"""
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        celim = lambda x: x < p
        A = ElimrG(data);
        r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ (-fracc(i|A|j, i|A|p), p, j) for j in filter(celim, range(1,A.n+1)) ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(p)                
        transElimIzda = transformaciones
        r = 0;  transformaciones = [];  colExcluida = set()
        for i in range(1,A.m+1):
            p = BuscaNuevoPivote(i|A); 
            if p:
                r += 1
                Tr = T([ (fracc(1, i|A|p), p) ])
                transformaciones += [Tr]  if Tr.t else []
                A & T( Tr )
                colExcluida.add(p)                
        pasos = [ [], A.pasos[1] + transElimIzda  + [T(transformaciones)] ]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix

class ElimF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma pre-escalonada de Matrix(data)

           operando con las filas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        A = Elim(~Matrix(data));     r = A.rango
        pasos = [ list(reversed([ ~t for t in A.pasos[1] ])), [] ]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(~A)
        self.__class__ = Matrix
        
class ElimGF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada de Matrix(data)

           operando con las filas (y evitando operar con fracciones). 
           Si rep es no nulo, se muestran en Jupyter los pasos dados"""
        A = ElimG(~Matrix(data));    r = A.rango
        pasos = [ list(reversed([ ~t for t in A.pasos[1] ])), [] ]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(~A)
        self.__class__ = Matrix
        
class ElimGJF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve una forma escalonada reducida de Matrix(data)

           operando con las columnas (y evitando operar con fracciones  
           hasta el último momento). Si rep es no nulo, se muestran en 
           Jupyter los pasos dados"""
        A = ElimGJ(~Matrix(data));   r = A.rango
        pasos = [ list(reversed([ ~t for t in A.pasos[1] ])), [] ]
        pasos = [ filtradopasos(pasos[i]) for i in (0,1) ]
        pasosPrevios = data.pasos if hasattr(data, 'pasos') and data.pasos else [[],[]]
        TexPasosPrev = data.tex   if hasattr(data, 'tex')   and data.tex   else []
        self.tex = rprElim(data, pasos, TexPasosPrev)
        pasos[0] = pasos[0] + pasosPrevios[0] 
        pasos[1] = pasosPrevios[1] + pasos[1]
        self.pasos = pasos 
        self.TrF = T(pasos[0])
        self.TrC = T(pasos[1])
        if rep:
            display(Math(self.tex))                                                               
        self.rango = r
        super(self.__class__ ,self).__init__(~A)
        self.__class__ = Matrix
        
def rprElim(data, pasos, TexPasosPrev=[]):
    """Escribe en LaTeX los pasos efectivos y las sucesivas matrices"""
    A = data.copy()
    if isinstance (data, Matrix):
        A.cF, A.cC = data.cF, data.cC
        
    tex = latex(data) if not TexPasosPrev else TexPasosPrev
    for l in 0,1:
        if l==0:
            for i in reversed(range(len(pasos[l]))):
                tex += '\\xrightarrow[' + latex(pasos[l][i]) + ']{}' 
                tex += latex( pasos[l][i] & A )
        if l==1:
            for i in range(len(pasos[l])):
                tex += '\\xrightarrow{' + latex(pasos[l][i]) + '}'
                tex += latex( A & pasos[l][i] )
                                                               
    return tex

def rprElimFyC(data, pasos, TexPasosPrev=[]):
    """Escribe en LaTeX los pasos efectivos y las sucesivas matrices"""
    A = data.copy()
    if isinstance (data, Matrix):
        A.cF, A.cC = data.cF, data.cC

    #pasos[0] = list(reversed(pasos[0]))
                                                               
    tex = latex(data) if not TexPasosPrev else TexPasosPrev
    for i in range(len(pasos[1])):
        tex += '\\xrightarrow' \
                + '[' + latex(T(pasos[0][-i-1])) + ']' \
                + '{' + latex(T(pasos[1][i])) + '}'
        tex += latex( pasos[0][-i-1] & A & pasos[1][i] )
                                                               
    return tex

def rprElimCF(data, pasos, TexPasosPrev=[]):
    """Escribe en LaTeX los pasos efectivos y las sucesivas matrices"""
    A = data.copy()
    if isinstance (data, Matrix):
        A.cF, A.cC = data.cF, data.cC
                                                               
    #pasos[0] = list(reversed(pasos[0]))
                                                               
    tex = latex(data) if not TexPasosPrev else TexPasosPrev
    for i in range(len(pasos[1])):
        tex += '\\xrightarrow[]{' + latex(T(pasos[1][i])) + '}'
        tex += latex( A & pasos[1][i] )
        tex += '\\xrightarrow['   + latex(T(pasos[0][-i-1])) + ']{}' 
        tex += latex( pasos[0][-i-1] & A )
                                                               
    return tex

def dispElim(self, pasos, TexPasosPrev=[]):
    display(Math(rprElim(self, pasos, TexPasosPrev)))

def dispElimFyC(self, pasos, TexPasosPrev=[]):
    display(Math(rprElimFyC(self, pasos, TexPasosPrev)))

def dispElimCF(self, pasos, TexPasosPrev=[]):
    display(Math(rprElimCF(self, pasos, TexPasosPrev)))

class InvMat(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las columnas"""
        A          = Matrix(data)        
        if not A.es_cuadrada():  raise ValueError('Matrix no cuadrada')
        R          = ElimGJ(A)
        self.pasos = R.pasos 
        self.TrF   = R.TrF 
        self.TrC   = R.TrC 
        self.tex   = rprElim( A.apila( I(A.n), 1 ) , self.pasos)
        if R.rango < A.n:        raise ArithmeticError('Matrix singular')        
        Inversa    = I(A.n) & T(R.pasos[1])  
        super(self.__class__ ,self).__init__(Inversa)
        self.__class__ = Matrix
        if rep:
            display(Math(self.tex))

class InvMatF(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas"""
        A          = Matrix(data)
        if A.m != A.n:
            raise ValueError('Matrix no cuadrada')
        M          = ElimGJF(A)
        self.pasos = M.pasos 
        self.TrF   = M.TrF 
        self.TrC   = M.TrC 
        self.tex   = rprElim( A.concatena(I(A.m),1) , self.pasos)
        if M.rango < A.n:
            raise ArithmeticError('Matrix singular')        
        Inversa    = T(M.pasos[0]) & I(A.n)   
        super(self.__class__ ,self).__init__(Inversa)
        self.__class__ = Matrix
        if rep:
            display(Math(self.tex))

class InvMatFC(Matrix):
    def __init__(self, data, rep=0):
        """Devuelve la matriz inversa y los pasos dados sobre las filas y columnas"""
        A          = Matrix(data)
        if A.m != A.n:
            raise ValueError('Matrix no cuadrada')
        M          = ElimGJ(ElimGF(A))
        self.pasos = M.pasos  
        self.TrF   = M.TrF 
        self.TrC   = M.TrC 
        self.tex   = rprElim( \
                     A.apila(I(A.n),1).concatena(I(A.n).apila(M0(A.n,A.n),1),1), \
                              self.pasos)
        if M.rango < A.n:
            raise ArithmeticError('Matrix singular')        
        Inversa    = ( I(A.n) & T(M.pasos[1]) ) * ( T(M.pasos[0]) & I(A.n) )
        super(self.__class__ ,self).__init__(Inversa)
        self.__class__ = Matrix
        if rep:
            display(Math(self.tex))

class SubEspacio:
    def __init__(self,data):
        """Inicializa un SubEspacio de Rn"""
        def SGenENulo(A):
            """Encuentra un sistema generador del Espacio Nulo de A"""
            K = Elim(A);   E = I(A.n) & T(K.pasos[1])
            S = Sistema([ v for j, v in enumerate(E,1) if (K|j).es_nulo() ])
            return S if S else Sistema([V0(A.n)])
        if not isinstance(data, (Sistema, Matrix)):
            raise ValueError(' Argumento debe ser un Sistema o Matrix ')
        if isinstance(data, Sistema):
            A          = Matrix(data)
            self.base  = Sistema([ c for c in Elim(A) if c.no_es_nulo() ])
            self.dim   = len(self.base)
            self.sgen  = self.base if self.base else Sistema([ V0(A.m) ])
            self.cart  = ~Matrix(SGenENulo(~A))
            self.Rn    = A.m
        if isinstance(data, Matrix):
            A          = data
            self.sgen  = SGenENulo(A)  
            self.dim   = 0 if self.sgen.es_nulo() else len(self.sgen)
            self.base  = self.sgen if self.dim else Sistema([])
            self.cart  = ~Matrix(SGenENulo(~Matrix(self.sgen)))
            self.Rn    = A.n
    def contenido_en(self, other):
        """Indica si este SubEspacio está contenido en other"""
        self.verificacion(other)
        if isinstance(other, SubEspacio):
            return all ([ (other.cart*v).es_nulo() for v in self.sgen ])
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
        return SubEspacio(Sistema(self.sgen.concatena(other.sgen)))

    def __and__(self, other):
        """Devuelve la intersección de subespacios"""
        self.verificacion(other)
        if isinstance(other, SubEspacio):
            return SubEspacio( self.cart.apila(other.cart) )
        elif  isinstance(other, EAfin):
            return other & self

    def __invert__(self):
        """Devuelve el complemento ortogonal"""
        return SubEspacio( Sistema( ~(self.cart) ) )

    def __contains__(self, other):
        """Indica si un Vector pertenece a un SubEspacio"""
        if not isinstance(other, Vector) or other.n != self.cart.n:
            raise ValueError\
                  ('El Vector no tiene el número adecuado de componentes')
        return (self.cart*other == V0(self.cart.m))

    def _repr_html_(self):
        """Construye la representación para el entorno Jupyter Notebook"""
        return html(self.latex())

    def EcParametricas(self, d=0):
        """Representación paramétrica del SubEspacio"""
        if d: display(Math(self.EcParametricas()))
        return EAfin(self.sgen,self.sgen|1).EcParametricas()

    def EcCartesianas(self, d=0):
        """Representación cartesiana del SubEspacio"""
        if d: display(Math(self.EcCartesianas()))
        return EAfin(self.sgen,self.sgen|1).EcCartesianas()
        
    def latex(self):
        """ Construye el comando LaTeX para un SubEspacio de Rn"""
        return EAfin(self.sgen,self.sgen|1).latex()
            

class EAfin:
    def __init__(self, data, v, vi=0):
        """Inicializa un Espacio Afín de Rn"""
        self.S  = data if isinstance(data, SubEspacio) else SubEspacio(data)
        if not isinstance(v, Vector) or v.n != self.S.Rn:
             raise ValueError('v y SubEspacio deben estar en el mismo espacio vectorial')
        self.v  = Vector(v) if vi else Elim( self.S.sgen.concatena(Sistema([v])) )|0
        self.Rn = self.S.Rn
        
    def __contains__(self, other):
        """Indica si un Vector pertenece a un EAfin"""
        if not isinstance(other, Vector) or other.n != self.S.cart.n:
            raise ValueError('Vector con un número inadecuado de componentes')
        return (self.S.cart)*other == (self.S.cart)*self.v

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
        self.verificacion(other)
        return not (self == other)

    def verificacion(self,other):
        if not isinstance(other, (SubEspacio, EAfin)) or  not self.Rn == other.Rn: 
            raise \
             ValueError('Ambos argumentos deben ser subconjuntos de en un mismo espacio')

    def __and__(self, other):
        """Devuelve la intersección de este EAfin con other"""
        self.verificacion(other)
        if isinstance(other, EAfin):
            M = self.S.cart.apila( other.S.cart )
            w = (self.S.cart*self.v).concatena( other.S.cart*other.v )
        elif isinstance(other, SubEspacio):
            M = self.S.cart.apila( other.cart )
            w = (self.S.cart*self.v).concatena( V0(other.cart.m) )                                                      
        try:
            S=SEL(M,w)
        except:
            print('Intersección vacía')
            return Sistema([])
        else:
            return S.eafin

    def __invert__(self):
        """Devuelve el mayor SubEspacio perpendicular a self"""
        return SubEspacio( Sistema( ~(self.S.cart) ) )

    def _repr_html_(self):
        """Construye la representación para el entorno Jupyter Notebook"""
        return html(self.latex())

    def EcParametricas(self, d=0):
        """Representación paramétrica de EAfin"""
        punto = latex(self.v) + '+' if (self.v != 0*self.v) else ''
        if d: display(Math(self.EcParametricas()))
        return '\\left\\{ \\boldsymbol{v}\\in\\mathbb{R}^' \
          + latex(self.S.Rn) \
          + '\ \\left|\ \\exists\\boldsymbol{p}\\in\\mathbb{R}^' \
          + latex(max(self.S.dim,1)) \
          + ',\\; \\boldsymbol{v}= ' \
          + punto \
          + latex(Matrix(self.S.sgen)) \
          + '\\boldsymbol{p}\\right. \\right\\}' \

    def EcCartesianas(self, d=0):
        """Representación cartesiana de EAfin"""
        if d: display(Math(self.EcCartesianas()))
        return '\\left\\{ \\boldsymbol{v}\\in\\mathbb{R}^' \
          + latex(self.S.Rn) \
          + '\ \\left|\ ' \
          + latex(self.S.cart) \
          + '\\boldsymbol{v}=' \
          + latex(self.S.cart*self.v) \
          + '\\right.\\right\\}' \
        
    def latex(self):
        """ Construye el comando LaTeX para un EAfin de Rn"""
        return self.EcParametricas() + '\\; = \\;' + self.EcCartesianas()
            
class Homogenea:
    def __init__(self, data, rep=0):
        """Resuelve un Sistema de Ecuaciones Lineales Homogéneo
    
        y muestra los pasos para encontrarlo"""
        
        A     = Matrix(data)
        L     = Elim( A )  
        E     = I(A.n) & T(L.pasos[1])
        base  = [ v for j, v in enumerate(E, 1) if (L|j).es_nulo() ]
        
        self.sgen        = Sistema(base) if base else Sistema([ V0(A.n) ])
        self.determinado = (len(base) == 0)
        self.pasos       = L.pasos; 
        self.TrF         = L.TrF 
        self.TrC         = L.TrC
        self.tex         = rprElim( A.apila( I(A.n) ,1 ) , self.pasos)
        self.enulo       = SubEspacio(self.sgen)
        
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
    def __init__(self, A, b, rep=0):
        """Resuelve un Sistema de Ecuaciones Lineales

        mediante eliminación por columnas en la matriz ampliada y muestra
        los pasos dados"""
        A  = Matrix(A)
        MA = A.concatena(Matrix([-b])).apila(I(A.n+1))
        MA.cfil( {A.m, A.m+A.n} ).ccol( {A.n} )
        
        L  = Elim( slice(1,A.m)|MA )
        
        EA        = Matrix(MA) & T(L.pasos[1]) 
        Normaliza = T([]) if (0|EA|0)==1 else T([( fracc(1,0|EA|0), EA.n )])
        EA & Normaliza

        K =         slice(1,A.m)|EA|slice(1,A.n);
        E = slice(A.m+1,A.m+A.n)|EA|slice(1,A.n)   
        S = slice(A.m+1,A.m+A.n)|EA|slice(A.n+1,None)  

        self.base        = Sistema([ v for j, v in enumerate(E,1) if (K|j).es_nulo() ])
        self.sgen        = self.base if self.base else Sistema([ V0(A.n) ])
        self.determinado = (len(self.base) == 0)

        if (L|0).no_es_nulo():
            self.solP  = set()
            self.eafin = set()
        else:
            self.solP  = S|1 
            self.eafin = EAfin(self.sgen, self.solP, 1)

        self.pasos       = [[], L.pasos[1]+[Normaliza] ] if Normaliza.t else [[], L.pasos[1]]
        self.TrF         = T(self.pasos[0]) 
        self.TrC         = T(self.pasos[1]) 
        self.tex         = rprElim( MA, self.pasos )
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
            return self.eafin.EcParametricas() if self.solP else latex(set())

              
class Determinante:
    def __init__(self, data, disp=0):
        """Calcula el determinante

        mediante eliminación Gaussiana por columnas y muestra los pasos dados"""
        
        A  = Matrix(data)
        
        if not A.es_cuadrada():  raise ValueError('Matrix no cuadrada')
        
        def calculoDet(A):
            producto  = lambda x: 1 if not x else x[0] * producto(x[1:])
            
            pc  = (A.L().pasos[1])
            ME  = A.extDiag(I(1),1)
            tex = ''
            pasos = [[],[]]
            
            for i in range(len(pc)):
                S  = [ tr for tr in filter( lambda x: len(x)==2, T(pc[i]).t ) ]
                m  = [-1 if isinstance(tr,set) else tr[0] for tr in S]
                pf = [T([ ( fracc(1, producto(m)) , A.n+1 ) ]) if producto(m)!=1 else T([])]
                
                tex = rprElimFyC(ME,[pf,[pc[i]]],tex)
                
                T(pf) & ME & T(pc[i])
                
                pasos[0] = pf + pasos[0]
                pasos[1] = pasos[1] + [pc[i]]
                
            Det = simplifica( producto( ME.diag() ) )
            
            return [tex, Det, pasos]
            
        
        self.tex, self.valor, self.pasos = calculoDet( A )
        
        self.TrF   = T(self.pasos[0])
        self.TrC   = T(self.pasos[1])
        
        if disp:
           display(Math(self.tex))
        
    def __repr__(self):
        """ Muestra un Sistema en su representación Python """
        return 'Valor del determinante:  ' + repr (self.valor) 

    def _repr_html_(self):
        """ Construye la representación para el entorno Jupyter Notebook """
        return html(self.latex())

    def latex(self):
        """ Construye el comando LaTeX para representar un Sistema """
        return latex(self.valor)


class DiagonalizaS(Matrix):
    def __init__(self, A, espectro, Rep=0):
        """Diagonaliza por bloques triangulares una Matrix cuadrada 

        Encuentra una matriz diagonal semejante mediante trasformaciones de sus
        columnas y las correspondientes transformaciones inversas espejo de las
        filas. Requiere una lista de autovalores (espectro), que deben aparecer
        en dicha lista tantas veces como sus respectivas multiplicidades 
        algebraicas. Los autovalores aparecen en la diagonal principal de la 
        matriz diagonal. El atributo S de dicha matriz diagonal es una matriz 
        cuyas columnas son autovectores de los correspondientes autovalores.
        """
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        D            = Matrix(A)
        if not D.es_cuadrada: raise ValueError('Matrix no es cuadrada')
        if not isinstance(espectro, list):
            raise ValueError('espectro no es una lista')
        if len(espectro)!=D.n:
            raise ValueError('número inadecuado de autovalores en la lista espectro')
        S            = I(D.n)
        Tex          = latex( D.apila(S,1) )
        pasosPrevios = [[],[]]
        selecc       = list(range(1,D.n+1))
        for lamda in espectro:
            m = selecc[-1]
            D = D-(lamda*I(D.n))
            Tex += '\\xrightarrow[' + latex(lamda) + '\\mathbf{I}]{(-)}' \
                                    + latex(D.apila(S,1))
            TrCol = filtradopasos(ElimG(selecc|D|selecc).pasos[1])
            pasos           = [ [], TrCol ]
            pasosPrevios[1] = pasosPrevios[1] + pasos[1]

            Tex = rprElim( D.apila(S,1), pasos, Tex) if TrCol else Tex
            D = D & T(pasos[1])
            S = S & T(pasos[1])

            pasos           = [ [T(pasos[1]).espejo()**-1] , []]
            pasosPrevios[0] = pasos[0] + pasosPrevios[0]

            Tex = rprElim( D.apila(S,1), pasos, Tex) if TrCol else Tex
            D   = T(pasos[0]) & D
            if m < D.n:
                transf = []; colExcluida = set(selecc)
                for i in range(m,D.n+1):
                    p = BuscaNuevoPivote(i|D);
                    if p:
                        TrCol = filtradopasos([ T([(-fracc(i|D|m, i|D|p), p, m)]) ])
                        pasos           = [ [], TrCol ]
                        pasosPrevios[1] = pasosPrevios[1] + pasos[1]

                        Tex = rprElim( D.apila(S,1), pasos, Tex) if TrCol else Tex
                        D = D & T(pasos[1])
                        S = S & T(pasos[1])

                        pasos           = [ [T(pasos[1]).espejo()**-1] , []]
                        pasosPrevios[0] = pasos[0] + pasosPrevios[0]

                        Tex = rprElim( D.apila(S,1), pasos, Tex) if TrCol else Tex
                        D   = T(pasos[0]) & D
                        colExcluida.add(p)                        
            D = D+(lamda*I(D.n))
            Tex += '\\xrightarrow[' + latex(lamda) + '\\mathbf{I}]{(+)}' \
                                    + latex(D.apila(S,1))
            
            selecc.pop()
            
        if Rep:
            display(Math(Tex))
            
        espectro.sort(reverse=True)                
        self.espectro = espectro
        self.tex = Tex
        self.S   = S
        self.TrF = T(pasosPrevios[0])
        self.TrC = T(pasosPrevios[1])
        self.pasos = pasosPrevios
        super(self.__class__ ,self).__init__(D)
        self.__class__ = Matrix
                   
class DiagonalizaO(Matrix):
    def __init__(self, A, espectro, Rep=0):
        """ Diagonaliza ortogonalmente una Matrix simétrica 

        Encuentra una matriz diagonal por semejanza empleando una matriz
        ortogonal Q a la derecha y su inversa (transpuesta) por la izquierda. 
        Requiere una lista de autovalores (espectro), que deben aparecer tantas
        veces como sus respectivas multiplicidades algebraicas. Los autovalores 
        aparecen en la diagonal principal de la matriz diagonal. El atributo Q 
        de la matriz diagonal es la matriz ortogonal cuyas columnas son 
        autovectores de los correspondientes autovalores. """
        def BaseOrtNor(q):
            "Crea una base ortonormal cuyo último vector es 'q'"
            if not isinstance(q,Vector): raise ValueError('El argumento debe ser un Vector')
            M = Matrix([q]).concatena(I(q.n)).GS()
            l = [ j for j, v in enumerate(M, 1) if v.no_es_nulo() ]
            l = l[1:len(l)]+[l[0]]
            return (M|l).normalizada()
           
        D =Matrix(A)
        if not D.es_simetrica:
           raise ValueError('La matriz no es simétrica')
        if not isinstance(espectro,list) or len(espectro)!=A.n:
           raise ValueError('Espectro incorrecto')
           
        S        = I(A.n)
        espectro = list(espectro);
        selecc   = list(range(1,D.n+1))
        for l in espectro:
            D = D - l*I(D.n)
            TrCol = ElimG(selecc|D|selecc).pasos[1]
            D = D + l*I(D.n)
            k       = len(selecc)
            nmenosk = (D.n)-k
            selecc.pop()

            q = ( I(k) & T(TrCol) )|0
            q = (sympy.sqrt(q*q)) * q
           
            Q = BaseOrtNor(q).concatena(M0(k,nmenosk)).apila( \
                M0(nmenosk,k).concatena(I(nmenosk)))  if nmenosk else BaseOrtNor(q)
           
            S = S *Q     
            D = ~Q*D*Q
            
        self.Q = S
        espectro.sort(reverse=True)                
        self.espectro = espectro
        super(self.__class__ ,self).__init__(D)
        self.__class__ = Matrix
                   
class DiagonalizaC(Matrix):
    def __init__(self, data, Rep=0):
        """ Diagonaliza por congruencia una Matrix simétrica (evitando dividir)

        Encuentra una matriz diagonal por conruencia empleando una matriz B 
        invertible (y entera si es posible) por la derecha y su transpuesta por
        la izquierda. No emplea los autovalores. En general los elementos en la
        diagonal principal no son autovalores, pero hay tantos elementos 
        positivos en la diagonal como autovalores positivos, tantos negativos 
        como autovalores negativos, y tantos ceros como auntovalores nulos. """
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        A     = Matrix(data);      colExcluida  = set()
        celim = lambda x: x > p;   pasosPrevios = [ [], [] ]
        #Tex   = latex(A);   
        for i in range(1,A.n):
            p = BuscaNuevoPivote(i|A)
            j = [k for k,col in enumerate(A|slice(i,None),i) if (i|col and not k|col)]
            if not (i|A|i):
                if j:
                    Tr = T( (1, j[0], i) )
                    p = i
                    pasos = [ [], filtradopasos([Tr]) ]
                    pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                    A = A & T(pasos[1])

                    pasos = [ filtradopasos([~Tr]) , []]
                    pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                    A = T(pasos[0]) & A
                elif p:
                    Tr = T( {i, p} )
                    p = i
                    pasos = [ [], filtradopasos([Tr]) ]
                    pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                    A = A & T(pasos[1])

                    pasos = [ filtradopasos([~Tr]) , []]
                    pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                    A = T(pasos[0]) & A
            if p:
                Tr = T( [ T( [ ( denom((i|A|j),(i|A|p)),    j),    \
                               (-numer((i|A|j),(i|A|p)), p, j)  ] ) \
                                              for j in filter(celim, range(1,A.n+1)) ] )
                pasos = [ [], filtradopasos([Tr]) ]
                pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                A = A & T(pasos[1])

                pasos = [ filtradopasos([~Tr]) , []]
                pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                A = T(pasos[0]) & A
            colExcluida.add(i)
           
        self.pasos     = pasosPrevios
        self.tex       = rprElimCF(Matrix(data),self.pasos) 
        self.TrF       = filtradopasos(T(self.pasos[0]))
        self.TrC       = filtradopasos(T(self.pasos[1]))
        self.B         = I(A.n) & self.TrC
        
        if Rep: 
            display(Math(self.tex))
            
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix
                   
class DiagonalizaCr(Matrix):
    def __init__(self, data, Rep=0):
        """ Diagonaliza por congruencia una Matrix simétrica

        Encuentra una matriz diagonal congruente multiplicando por una matriz
        invertible B a la derecha y por la transpuesta de B por la izquierda. 
        No requiere conocer los autovalores. En general los elementos en la
        diagonal principal de la matriz diagonal no son autovalores, pero hay
        tantos elementos positivos en la diagonal como autovalores positivos
        (incluyendo la multiplicidad de cada uno), tantos negativos como
        autovalores negativos (incluyendo la multiplicidad de cada uno), y tantos
        ceros como la multiplicidad algebraica del autovalor cero. """
        def BuscaNuevoPivote(self, r=0):
            ppivote = lambda v, k=0:\
                      ( [i for i,c in enumerate(v, 1) if (c!=0 and i>k)] + [0] )[0]
            pp = ppivote(self, r)
            while pp in colExcluida:
                pp = ppivote(self, pp)
            return pp
        A     = Matrix(data);      colExcluida  = set()
        celim = lambda x: x > p;   pasosPrevios = [ [], [] ]
        Tex   = latex(A);   
        for i in range(1,A.n):
            p = BuscaNuevoPivote(i|A)
            j = [k for k,col in enumerate(A|slice(i,None),i) if (i|col and not k|col)]
            if not (i|A|i):
                if j:
                    Tr = T( (1, j[0], i) )
                    p = i
                    pasos = [ [], filtradopasos([Tr]) ]
                    pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                    A = A & T(pasos[1])

                    pasos = [ filtradopasos([~Tr]) , []]
                    pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                    A = T(pasos[0]) & A
                elif p:
                    Tr = T( {i, p} )
                    p = i
                    pasos = [ [], filtradopasos([Tr]) ]
                    pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                    A = A & T(pasos[1])

                    pasos = [ filtradopasos([~Tr]) , []]
                    pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                    A = T(pasos[0]) & A
            if p:
                Tr = T([ (-fracc(i|A|j, i|A|p), p, j) for j in filter(celim, range(1,A.n+1)) ])
                pasos = [ [], filtradopasos([Tr]) ]
                pasosPrevios[1] = pasosPrevios[1] + pasos[1]
                A = A & T(pasos[1])

                pasos = [ filtradopasos([~Tr]) , []]
                pasosPrevios[0] = pasos[0] + pasosPrevios[0]
                A = T(pasos[0]) & A
            colExcluida.add(i)
            
        self.tex       = Tex
        self.pasos     = pasosPrevios
        self.TrF       = T(self.pasos[0])
        self.TrC       = T(self.pasos[1])
        self.B         = I(A.n) & T(pasosPrevios[1])
        
        if Rep:
            display(Math(Tex))
            
        super(self.__class__ ,self).__init__(A)
        self.__class__ = Matrix
        
