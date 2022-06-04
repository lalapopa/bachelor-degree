SHELL := bash

.PHONY: compose
compose:
	cp main.pdf -t ./compose;
	cp ../special_part/forms/form.pdf -t ./compose; 
	cp ../presentation/slides.pdf ./compose/Слайды-ВКР-Москвитин-А.pdf;
	qpdf ./compose/form.pdf --pages . 1-5 -- ./compose/Формы-Москвитин.pdf;
	qpdf ./compose/form.pdf --pages . 3 ./compose/main.pdf -- ./compose/ВКР-Москвитин-А.pdf;
	rm ./compose/{main,form}.pdf;



