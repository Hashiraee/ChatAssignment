SHELL=/bin/bash
VIRTUALENV?=env
.PHONY: help install run clean envclean calculate

help:
	@echo "Make targets:"
	@echo " install     Create virtual environment (env) and install required packages"
	@echo " freeze      Persist installed packages to requirements.txt"
	@echo " clean       Remove *.pyc files and __pycache__ directory"
	@echo " purge       Remove storage files"
	@echo " envclean    Remove virtual environment (env)"
	@echo " run         Run the code"
	@echo " agent       Run the code with the agent"
	@echo " calculate   Run the produced code"
	@echo "Check the Makefile for more details"

install:
	@python3 -m venv $(VIRTUALENV)
	@. $(VIRTUALENV)/bin/activate; pip3 install --upgrade pip; pip3 install -r requirements.txt

freeze:
	@. $(VIRTUALENV)/bin/activate; pip3 freeze > requirements.txt

clean:
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} \+

purge:
	@rm -rf ./storage

envclean: clean
	@rm -rf $(VIRTUALENV)

calculate:
	@python3 output/math_expression.py

run:
	@. $(VIRTUALENV)/bin/activate; python3 code/chat.py "$(filter-out $@,$(MAKECMDGOALS))"

%:
	@:
