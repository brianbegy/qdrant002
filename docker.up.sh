#!/bin/zsh
docker pull qdrant/qdrant
docker stop qdrant_local || true
docker rm qdrant_local || true
docker run --name qdrant_local -d -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/data/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

