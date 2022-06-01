SHELL := bash

.PHONY: compose
compose:
	cp main.pdf -t ./compose;
	cp ../special_part/forms/form.pdf -t ./compose; 
	cp ../presentation/slides.pdf ./compose/Слайды-ВКР-Москвитин-А.pdf;
	qpdf ./compose/form.pdf --pages . 3 -- ./compose/title.pdf;
	qpdf --empty --pages ./compose/title.pdf ./compose/main.pdf -- ./compose/ВКР-Москвитин-А.pdf;
	rm ./compose/{title,main,form}.pdf;



