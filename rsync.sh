#!/bin/bash

# Sync local changes to the remote build/run hosts on every file change.
#
# watchexec runs its command through a shell already, so we pass the whole
# host loop as ONE command string (not `sh -c '...'`, which double-wraps and
# breaks the inner -c). `$h` is single-quoted on purpose: it must expand in
# watchexec's shell when the command runs, not in this script — so the
# SC2016 shellcheck warning here is expected.
#
# shellcheck disable=SC2016
watchexec --debounce 300ms -- 'for h in gpu vaes; do
  rsync -az --delete --exclude=target/ --exclude=.git/ --exclude=.env --exclude=CLAUDE.local.md ./ "$h:~/rust-cuda-aes/"
done'
