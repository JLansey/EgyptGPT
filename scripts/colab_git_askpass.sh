#!/usr/bin/env bash

case "${1:-}" in
  *Username*)
    printf "%s" "x-access-token"
    ;;
  *Password*)
    printf "%s" "${GITHUB_TOKEN:-}"
    ;;
  *)
    printf "\n"
    ;;
esac
