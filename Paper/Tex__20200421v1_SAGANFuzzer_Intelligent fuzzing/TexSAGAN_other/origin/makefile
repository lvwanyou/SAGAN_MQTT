mainfile = Main_B
all:
	pdflatex $(mainfile) && bibtex $(mainfile) && pdflatex $(mainfile) && pdflatex $(mainfile)
clean:
	rm *.aux \
	*.fls \
	*.log \
	*.out \
	*.bbl \
	*.blg \
	*.fdb_latexmk \
	*.synctex.gz