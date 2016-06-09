#!/bin/bash

if hash git 2>/dev/null; then
	git describe --tags --always 2>/dev/null && exit 0
fi
echo "v0.01-?"
