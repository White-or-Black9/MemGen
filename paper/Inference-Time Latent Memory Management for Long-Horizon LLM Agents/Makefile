PDF=main.pdf

all: $(PDF)

$(PDF): main.tex sections/*.tex tables/*.tex references.bib
	latexmk -pdf main.tex

clean:
	latexmk -C
