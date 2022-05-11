.PHONY: all say_hello generate clean
all: copyfiles

SHELL := /bin/bash

FILES_TABLE:=./code_output/RESULTS
FILES_FIGURES:=./code_output/PLOTS_PGF
copyfiles:
	cp -r $(FILES_TABLE)/*.tex ./report/tables
	cp -r $(FILES_FIGURES)/*.pgf ./report/figures


