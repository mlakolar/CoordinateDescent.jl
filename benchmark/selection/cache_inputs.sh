#!/bin/bash
# Reuse installed inputs without exposing another task's mutable compiled cache.
: "${RUN:?RUN must name the task run directory}"
mkdir -p "$RUN/shared-inputs"
for name in packages artifacts registries clones; do
  if [[ -d /home1/mkolar/juliaPkg/depot/$name && ! -e $RUN/shared-inputs/$name ]]; then
    ln -s "/home1/mkolar/juliaPkg/depot/$name" "$RUN/shared-inputs/$name"
  fi
done
