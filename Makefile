
lint:
	flake8 ./sayteacher

isort:
	isort -c --diff ./sayteacher

fisort:
	isort ./sayteacher

test:
	python -m pytest -vv ./sayteacher
	cargo test
