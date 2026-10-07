# source ./venv/bin/activate

lint:
	flake8 ./sayteacher ./transcreenpt

isort:
	isort -c --diff ./sayteacher ./transcreenpt

fisort:
	isort ./sayteacher ./transcreenpt

test:
	python -m pytest -vv ./sayteacher
	python -m pytest -vv ./transcreenpt
	cargo test
