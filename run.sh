#!/bin/bash

# Export current user's UID and GID
export UID=$(id -u)
export GID=$(id -g)

# Create data directory with proper permissions
mkdir -p data
chmod 777 data

# Run docker-compose
docker-compose up --build