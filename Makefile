# source ./venv/bin/activate

DATATRAFFIC_DIR := vendor/datatraffic
PKG_STAGE_DIR   := dist

lint:
	flake8 ./sayteacher ./transcreenpt

isort:
	isort -c --diff ./sayteacher ./transcreenpt

fisort:
	isort ./sayteacher ./transcreenpt

test:
	python -m pytest -vv ./sayteacher
	python -m pytest -vv ./transcreenpt
	cd microagent && cargo test
	cd microagent-transport && cargo test
	cd outvoiceagent && cargo test

# Build the pydatatraffic wheel in its own Docker builder (vendor/datatraffic/Dockerfile),
# then stage it into ./dist/ for the sayteacher image. Requires docker.
build-pydatatraffic:
	rm -rf $(PKG_STAGE_DIR) && mkdir -p $(PKG_STAGE_DIR)
	$(MAKE) -C $(DATATRAFFIC_DIR) dockerbuild
	cp /tmp/datatraffic/pydatatraffic-*.whl $(PKG_STAGE_DIR)/

# Build the sayteacher image (stages pydatatraffic first, then builds via docker compose).
docker-build: build-pydatatraffic
	docker compose build sayteacher-api

# Run the built container with GPU.
docker-run: docker-build
	docker compose up -d

.PHONY: lint isort fisort test build-pydatatraffic docker-build docker-run
