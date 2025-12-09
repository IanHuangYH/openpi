#!/bin/bash

SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.eval.yml up --build