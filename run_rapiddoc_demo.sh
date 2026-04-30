#!/bin/bash
# RapidDoc Gradio Demo Launcher
# This script automatically:
# 1. Activates virtual environment
# 2. Sets up environment variables
# 3. Opens file manager to test_files/ folder
# 4. Launches Gradio demo
# 5. Opens Chrome browser to http://0.0.0.0:7860

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  RapidDoc Gradio Demo Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Activate virtual environment
echo -e "${YELLOW}[1/5] Activating virtual environment...${NC}"
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${RED}✗ Virtual environment not found at venv/bin/activate${NC}"
    echo -e "${YELLOW}Please create a virtual environment first:${NC}"
    echo -e "  python -m venv venv"
    exit 1
fi

# Step 2: Set environment variables
echo -e "${YELLOW}[2/5] Setting up environment variables...${NC}"
if [ -f "deepx_scripts/set_env.sh" ]; then
    source deepx_scripts/set_env.sh 1 2 1 3 2 4
    echo -e "${GREEN}✓ Environment variables configured${NC}"
else
    echo -e "${RED}✗ deepx_scripts/set_env.sh not found${NC}"
    exit 1
fi

# Step 2.5: Check and kill existing process on port 7860
echo -e "${YELLOW}[2.5/6] Checking port 7860...${NC}"
PORT=7860

# Try lsof first
if command -v lsof &> /dev/null; then
    PID=$(lsof -ti:$PORT 2>/dev/null || true)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}⚠ Port $PORT is in use by process $PID${NC}"
        echo -e "${YELLOW}  Killing existing process...${NC}"
        kill -9 $PID 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}✓ Port $PORT is now free${NC}"
    else
        echo -e "${GREEN}✓ Port $PORT is available${NC}"
    fi
# Try netstat as fallback
elif command -v netstat &> /dev/null; then
    PID=$(netstat -tlnp 2>/dev/null | grep ":$PORT " | awk '{print $7}' | cut -d'/' -f1 || true)
    if [ ! -z "$PID" ] && [ "$PID" != "-" ]; then
        echo -e "${YELLOW}⚠ Port $PORT is in use by process $PID${NC}"
        echo -e "${YELLOW}  Killing existing process...${NC}"
        kill -9 $PID 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}✓ Port $PORT is now free${NC}"
    else
        echo -e "${GREEN}✓ Port $PORT is available${NC}"
    fi
# Try ss as another fallback
elif command -v ss &> /dev/null; then
    PID=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' || true)
    if [ ! -z "$PID" ]; then
        echo -e "${YELLOW}⚠ Port $PORT is in use by process $PID${NC}"
        echo -e "${YELLOW}  Killing existing process...${NC}"
        kill -9 $PID 2>/dev/null || true
        sleep 1
        echo -e "${GREEN}✓ Port $PORT is now free${NC}"
    else
        echo -e "${GREEN}✓ Port $PORT is available${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Cannot check port status (lsof/netstat/ss not found)${NC}"
    echo -e "${YELLOW}  Continuing anyway...${NC}"
fi

# Step 3: Open file manager to test_files/ folder
echo -e "${YELLOW}[3/6] Opening test_files/ folder...${NC}"
PDF_DIR="$SCRIPT_DIR/test_files"
if [ -d "$PDF_DIR" ]; then
    # Try different file managers (xdg-open, nautilus, dolphin, thunar, etc.)
    if command -v xdg-open &> /dev/null; then
        xdg-open "$PDF_DIR" &> /dev/null &
        echo -e "${GREEN}✓ File manager opened${NC}"
    elif command -v nautilus &> /dev/null; then
        nautilus "$PDF_DIR" &> /dev/null &
        echo -e "${GREEN}✓ Nautilus opened${NC}"
    elif command -v dolphin &> /dev/null; then
        dolphin "$PDF_DIR" &> /dev/null &
        echo -e "${GREEN}✓ Dolphin opened${NC}"
    elif command -v thunar &> /dev/null; then
        thunar "$PDF_DIR" &> /dev/null &
        echo -e "${GREEN}✓ Thunar opened${NC}"
    else
        echo -e "${YELLOW}⚠ File manager not found, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}⚠ test_files/ folder not found${NC}"
fi

# Step 4: Wait a moment for browser to be ready
echo -e "${YELLOW}[4/6] Preparing to launch browser...${NC}"
sleep 2

# Open browser in background
echo -e "${YELLOW}[5/6] Opening browser...${NC}"
URL="http://0.0.0.0:7860"

# Try to open Chrome/Chromium first, then Firefox, then default browser
if command -v google-chrome &> /dev/null; then
    google-chrome "$URL" &> /dev/null &
    echo -e "${GREEN}✓ Chrome opened${NC}"
elif command -v google-chrome-stable &> /dev/null; then
    google-chrome-stable "$URL" &> /dev/null &
    echo -e "${GREEN}✓ Chrome opened${NC}"
elif command -v chromium-browser &> /dev/null; then
    chromium-browser "$URL" &> /dev/null &
    echo -e "${GREEN}✓ Chromium opened${NC}"
elif command -v chromium &> /dev/null; then
    chromium "$URL" &> /dev/null &
    echo -e "${GREEN}✓ Chromium opened${NC}"
elif command -v firefox &> /dev/null; then
    firefox "$URL" &> /dev/null &
    echo -e "${GREEN}✓ Firefox opened${NC}"
elif command -v firefox-esr &> /dev/null; then
    firefox-esr "$URL" &> /dev/null &
    echo -e "${GREEN}✓ Firefox ESR opened${NC}"
elif command -v xdg-open &> /dev/null; then
    xdg-open "$URL" &> /dev/null &
    echo -e "${GREEN}✓ Default browser opened${NC}"
else
    echo -e "${YELLOW}⚠ Browser not found, you can manually open: ${URL}${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Starting Gradio Demo...${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}URL: ${URL}${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Step 6: Launch Gradio demo
python demo/gradio_app.py
