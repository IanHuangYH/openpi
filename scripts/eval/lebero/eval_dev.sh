#!/bin/bash

SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.dev.yml up -d --build
