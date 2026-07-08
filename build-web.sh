#!/bin/bash

# Exit on error
set -e

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define ANSI colors
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${CYAN}Building WebAssembly module...${NC}"
# Run wasm-pack build targeting web and outputting to the root docs/pkg directory
wasm-pack build "$SCRIPT_DIR/web" --target web --out-dir "$SCRIPT_DIR/docs/pkg"

echo -e "${CYAN}Cleaning up accidental/local duplicate outputs...${NC}"
DUPLICATE_PKG="$SCRIPT_DIR/web/pkg"
DUPLICATE_DOCS="$SCRIPT_DIR/web/docs"
REDUNDANT_PK="$SCRIPT_DIR/docs/pk"

if [ -d "$DUPLICATE_PKG" ]; then
    rm -rf "$DUPLICATE_PKG"
    echo -e "${YELLOW}Removed duplicate output at $DUPLICATE_PKG${NC}"
fi

if [ -d "$DUPLICATE_DOCS" ]; then
    rm -rf "$DUPLICATE_DOCS"
    echo -e "${YELLOW}Removed duplicate output at $DUPLICATE_DOCS${NC}"
fi

if [ -d "$REDUNDANT_PK" ]; then
    rm -rf "$REDUNDANT_PK"
    echo -e "${YELLOW}Removed redundant output at $REDUNDANT_PK${NC}"
fi

echo -e "${GREEN}Success! WASM build output is ready in docs/pkg/.${NC}"
