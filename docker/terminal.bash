#!/bin/bash
# WARNING! it is important to run with "bash -l" .. otherwise environmental variables from .profile won't get loaded
docker ps | grep -i "my-tensorflow-env" | awk '{print $1}' | xargs -I {} konsole --new-tab -e docker exec -it {} bash -l &
