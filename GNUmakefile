############################ -*- Mode: Makefile -*- ###########################
PAPER	   = NACAL
NWFuente   = NACAL_source
DOCU	   = Doc_$(PAPER).pdf
DIRFuente  = fuente

############
VPATH	   = $(PWD)/$(DIRFuente)
AUXFILES   = $(VPATH)/00Notacion.tex  $(VPATH)/$(NWFuente).nw # $(VPATH)/$(PAPER).bib 
DIRLOCAL   = $(subst $(HOME)/,$(empty),$(PWD))
DIRTEMP	   = $(HOME)/temporal/$(DIRLOCAL)
SHELL	   = /bin/bash

todo: programas documentacion 

documentacion: $(DIRTEMP) $(DIRTEMP)/$(SUBDIRSCRIPTS) $(VPATH)/$(NWFuente).nw programas
	make $(PAPER).pdf
#	cp $(PAPER).pdf ./doc/
#	make $(DOCU)

programas: $(DIRTEMP) $(DIRTEMP)/$(SUBDIRSCRIPTS) $(VPATH)/$(NWFuente).nw 
	make $(PYTHON)
	touch bin/__init__.py

############################## documentacion ###################################

NOWEAVE	 = noweave -index -delay -filter btdefn  # NOWEAVE  = noweave -delay -filter btdefn

NOTANGLE = notangle                              # NOTANGLE = nountangle -matlab

#### Regla para crear los .tex a partir de los .nw
%.tex:	%.nw 
	$(NOWEAVE) -delay $< > doc/$@

### Regla para obtener los .py a partir de los .nw del mismo nombre
%.py:	%.nw 
	$(NOTANGLE) -R$@ $<  > bin/$@


$(DIRTEMP): $(AUXFILES);
	mkdir -p bin/
	mkdir -p doc/
	mkdir -p doc/Notebooks/
	mkdir -p doc/Notebooks/TutorialPython/
	mkdir -p $@
	ln -s -f $? $@
	ln -s -f $(PWD)/bin                  $(DIRTEMP)
	ln -s -f $(VPATH)                    $(DIRTEMP)

clean: ; rm -r -f $(DIRTEMP); rm -f $(PAPER).pdf

############################## documentacion ###################################

$(PAPER).pdf:  $(DIRTEMP)  $(AUXFILES) #; #$(VPATH)/$(PAPER).tex 
	cd $(DIRTEMP); $(NOWEAVE) $(NWFuente).nw       >   $(DIRTEMP)/$(PAPER).tex
	rubber -d --into $(DIRTEMP) $(DIRTEMP)/$(PAPER).tex #$(<F)
	cp $(DIRTEMP)/$@ ./doc/
	#cp $(DIRTEMP)/$@ Notebooks/$(DOCU)

############################## programas ###################################

SUBDIRSCRIPTS = bin/

python  =			\
	nacal			\
	EjemploLiterateProgramming \
#	extension		\

##########################################################################################

PYTHON	:= $(addprefix $(DIRTEMP)/$(SUBDIRSCRIPTS), $(addsuffix .py,   $(python)))

$(PYTHON): $(DIRTEMP)/$(SUBDIRSCRIPTS) $(VPATH)/$(NWFuente).nw
	$(NOTANGLE) -R$(notdir $@) $(VPATH)/$(NWFuente).nw > $(DIRTEMP)/$(notdir $@)
	cp $(DIRTEMP)/$(notdir $@) bin 
	cp $(DIRTEMP)/$(notdir $@) doc/Notebooks
	cp $(DIRTEMP)/$(notdir $@) doc/Notebooks/TutorialPython

$(DIRTEMP)/$(SUBDIRSCRIPTS):
	mkdir -p  $@


##############################################################################

Commit:
	cd doc/Notebooks; git commit -a

Push:
	cd doc/Notebooks; git push -u origin master

Status:
	cd doc/Notebooks; git status

Add:
	cd doc/Notebooks; git add $(fichero)

# make Add fichero=Leccion04.ipynb 

####################### configuración local del git ##########################
# git config --list
##############################################################################

$(DOCU): $(DIRTEMP) $(AUXFILES) #$(NWFuente).nw
	cd $(DIRTEMP); $(NOWEAVE) $(NWFuente).nw       >   $(DIRTEMP)/$(PAPER).tex
	cd $(DIRTEMP); pdflatex -interaction errorstopmode $(DIRTEMP)/$(PAPER).tex
	cd $(DIRTEMP); pdflatex -interaction batchmode     $(DIRTEMP)/$(PAPER).tex
#	cd $(DIRTEMP); bibtex                              $(DIRTEMP)/$(NWFuente)    
	cd $(DIRTEMP); pdflatex -interaction batchmode     $(DIRTEMP)/$(PAPER).tex
	cd $(DIRTEMP); pdflatex -interaction batchmode     $(DIRTEMP)/$(PAPER).tex
	cp -f $(DIRTEMP)/$(PAPER).pdf $(DIRTEMP)/$(SUBDIRSCRIPTS)
	cp -f $(DIRTEMP)/$(SUBDIRSCRIPTS)$(PAPER).pdf $@ 
