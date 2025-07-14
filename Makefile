.PHONY: test

SHELL = /bin/sh

USER_ID := $(shell id -u)
GROUP_ID := $(shell id -g)

GPUS ?= '0'
MACHINE ?= default
CONFIG ?= ''
CHECKPOINT ?= 'best.ckpt'

RUN_IN_CONTAINER = docker --context $(MACHINE) compose run --remove-orphans -e CUDA_VISIBLE_DEVICES=$(GPUS)  3dlabgen 

build:
	COMPOSE_DOCKER_CLI_BUILD=1 docker compose build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID)

download:
	$(RUN_IN_CONTAINER) "bash -c ./download_assets.sh"

train:
	$(RUN_IN_CONTAINER) "CUDA_LAUNCH_BLOCKING=1 python train.py"

generate:
	$(RUN_IN_CONTAINER) "python generate.py --weights $(CHECKPOINT)"

compute_target:
	$(RUN_IN_CONTAINER) "python compute_target.py"

compute_gen_metrics:
	$(RUN_IN_CONTAINER) "python compute_generative_metrics.py"

shell:
	$(RUN_IN_CONTAINER) bash 

freeze_requirements:
	pip-compile requirements.in > requirements.txt

