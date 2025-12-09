#!/bin/bash

SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.clean.yml up --build