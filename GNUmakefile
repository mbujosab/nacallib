############################ -*- Mode: Makefile -*- ###########################
PAPER	   = NACAL
NWFuente   = NACAL_source
DOCU	   = Doc_$(PAPER).pdf
DIRFuente  = fuente

VERSION    = 0.1.30

############
VPATH	   = $(PWD)/$(DIRFuente)
AUXFILES   = $(VPATH)/00Notacion.tex  $(VPATH)/$(NWFuente).nw # $(VPATH)/$(PAPER).bib 
DIRLOCAL   = $(subst $(HOME)/,$(empty),$(PWD))
DIRTEMP	   = $(HOME)/temporal/$(DIRLOCAL)
SHELL	   = /bin/bash

todo: programas documentacion 

documentacion: $(DIRTEMP) $(DIRTEMP)/$(SUBDIRSCRIPTS) $(VPATH)/$(NWFuente).nw programas
#	make $(PAPER).pdf
#	cp $(PAPER).pdf ./doc/
	make $(DOCU)

programas: $(DIRTEMP) $(DIRTEMP)/$(SUBDIRSCRIPTS) $(VPATH)/$(NWFuente).nw 
	make $(PYTHON)

init:
	rm  -f nacal/__init__.py
	printf '""" \nnacal.\n\nNotación Asociativa para un curso de Álgebra Lineal (NAcAL).\n"""\n\n' >>  nacal/__init__.py
	printf '__version__ = "%s"\n' $(VERSION) >>  nacal/__init__.py
	printf '__author__  = "Marcos Bujosa"\n' >>  nacal/__init__.py
	printf '__name__    = "nacal"\n\n' >>  nacal/__init__.py
	printf 'from .nacal import *' >>  nacal/__init__.py


############################## documentacion ###################################

NOWEAVE	 = noweave -index -delay -filter btdefn  # NOWEAVE  = noweave -delay -filter btdefn

NOTANGLE = notangle                              # NOTANGLE = nountangle -matlab

#### Regla para crear los .tex a partir de los .nw
%.tex:	%.nw 
	$(NOWEAVE) -delay $< > doc/$@

### Regla para obtener los .py a partir de los .nw del mismo nombre
%.py:	%.nw 
	$(NOTANGLE) -R$@ $<  > nacal/$@


$(DIRTEMP): $(AUXFILES);
	mkdir -p nacal/
	mkdir -p doc/
	mkdir -p doc/Notebooks/
	mkdir -p $@
	ln -s -f $? $@
	ln -s -f $(PWD)/nacal                $(DIRTEMP)
	ln -s -f $(VPATH)                    $(DIRTEMP)

clean: ; rm -r -f $(DIRTEMP); rm -f $(PAPER).pdf

############################## documentacion ###################################

$(PAPER).pdf:  $(DIRTEMP)  $(AUXFILES) #; #$(VPATH)/$(PAPER).tex 
	cd $(DIRTEMP); $(NOWEAVE) $(NWFuente).nw       >   $(DIRTEMP)/$(PAPER).tex
	echo $(VERSION) > $(DIRTEMP)/version.txt
	rubber --unsafe -d --into $(DIRTEMP) $(DIRTEMP)/$(PAPER).tex #$(<F)
	cp $(DIRTEMP)/$@ ./doc/
	#cp $(DIRTEMP)/$@ Notebooks/$(DOCU)


#$(PAPER).pdf: $(DIRTEMP)  $(AUXFILES)
#	xelatex -interaction batchmode $(PAPER).tex
#	bib2gls --group $(PAPER)
#	pythontex --interpreter python:python3 $(PAPER).tex	
#	xelatex -interaction batchmode $(PAPER).tex
#	bib2gls --group $(PAPER)
#	latexmk -silent -pdflua $(PAPER)


############################## programas ###################################

SUBDIRSCRIPTS = nacal/

python  =			\
	nacal			\
	EjemploLiterateProgramming \
#	extension		\

##########################################################################################

PYTHON	:= $(addprefix $(DIRTEMP)/$(SUBDIRSCRIPTS), $(addsuffix .py,   $(python)))

$(PYTHON): $(DIRTEMP)/$(SUBDIRSCRIPTS) $(VPATH)/$(NWFuente).nw
	$(NOTANGLE) -R$(notdir $@) $(VPATH)/$(NWFuente).nw > $(DIRTEMP)/$(notdir $@)
	cp $(DIRTEMP)/$(notdir $@) nacal 
	#cp $(DIRTEMP)/$(notdir $@) doc/Notebooks
	#cp $(DIRTEMP)/$(notdir $@) doc/Notebooks/TutorialPython

$(DIRTEMP)/$(SUBDIRSCRIPTS):
	mkdir -p  $@


##############################################################################

Commit:
	git commit -a

Push:
	git push -u origin master

Status:
	git status

Add:
	git add $(fichero)

# make Add fichero=Leccion04.ipynb 

####################### configuración local del git ##########################
# git config --list
##############################################################################

$(DOCU): $(DIRTEMP) $(AUXFILES) #$(NWFuente).nw
	echo $(VERSION) > $(DIRTEMP)/version.txt
	cd $(DIRTEMP); $(NOWEAVE) $(NWFuente).nw       >   $(DIRTEMP)/$(PAPER).tex
#	cd $(DIRTEMP); pdflatex -interaction errorstopmode $(DIRTEMP)/$(PAPER).tex
	cd $(DIRTEMP); pdflatex -interaction batchmode     $(DIRTEMP)/$(PAPER).tex
	cd $(DIRTEMP); pythontex --interpreter python:python3 $(DIRTEMP)/$(PAPER).tex	
#	cd $(DIRTEMP); pdflatex -interaction errorstopmode $(DIRTEMP)/$(PAPER).tex
#	cd $(DIRTEMP); pdflatex -interaction batchmode     $(DIRTEMP)/$(PAPER).tex
#	cd $(DIRTEMP); bibtex                              $(DIRTEMP)/$(NWFuente)    
	cd $(DIRTEMP); pdflatex -interaction batchmode     $(DIRTEMP)/$(PAPER).tex
	cd $(DIRTEMP); pdflatex -interaction batchmode     $(DIRTEMP)/$(PAPER).tex
	cp -f $(DIRTEMP)/$(PAPER).pdf $(DIRTEMP)/$(SUBDIRSCRIPTS)
	cp -f $(DIRTEMP)/$(SUBDIRSCRIPTS)$(PAPER).pdf $@ 
	cp $(DIRTEMP)/$(PAPER).pdf ./doc/$(PAPER).pdf
