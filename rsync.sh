#!/bin/bash

# Used to sync local changes with gpu server where we run the code.
watchexec --debounce 300ms -- \
  rsync -az --delete \
  --exclude='target/' --exclude='.git/' --exclude='.env' --exclude='CLAUDE.local.md' \
  ./ gpu:~/rust-cuda-aes/
