PWEAVE_FLAGS=tex

corona: corona.ptexw corona.tex
    
corona.tex:
    pweave -f $(PWEAVE_FLAGS) corona.ptexw
    xelatex corona.tex
    bibtex corona
    xelatex corona
    xelatex corona
