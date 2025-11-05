#!/bin/bash
#
# Installation script for Jetson ML Platform Client
# For Jetson AGX Orin 32GB running JetPack 5.x/6.x
#

set -e

echo "============================================"
echo "Jetson ML Platform Client Installer"
echo "============================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: Please run as root (use sudo)"
    exit 1
fi

# Check if running on Jetson
if [ ! -f "/etc/nv_tegra_release" ]; then
    echo "WARNING: This does not appear to be a Jetson device"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Step 1: Creating directories..."
mkdir -p /opt/jetson-client/src
mkdir -p /opt/jetson-client/models
mkdir -p /opt/ml-platform/models
mkdir -p /etc/jetson-client
mkdir -p /var/log

echo "Step 2: Copying files..."
cp -r src/* /opt/jetson-client/src/
chmod +x /opt/jetson-client/src/jetson_client.py

# Copy config template if config doesn't exist
if [ ! -f "/etc/jetson-client/config.json" ]; then
    echo "Step 3: Creating configuration file..."
    cp config/config.json.template /etc/jetson-client/config.json
    echo "  -> Configuration created at: /etc/jetson-client/config.json"
    echo "  -> IMPORTANT: Edit this file with your backend URL!"
else
    echo "Step 3: Configuration file already exists, skipping..."
fi

echo "Step 4: Installing Python dependencies..."
pip3 install -r requirements.txt

# Optional: Install jetson-stats (recommended)
echo ""
read -p "Install jetson-stats for better GPU monitoring? (Y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo "Installing jetson-stats..."
    pip3 install jetson-stats
    echo "  -> jetson-stats installed"
    echo "  -> NOTE: You may need to reboot for jtop to work properly"
fi

echo ""
echo "Step 5: Installing systemd service..."
cp jetson-client.service /etc/systemd/system/
systemctl daemon-reload

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit configuration:"
echo "   sudo nano /etc/jetson-client/config.json"
echo ""
echo "2. Update the 'backend_url' to point to your server"
echo ""
echo "3. Start the service:"
echo "   sudo systemctl start jetson-client"
echo ""
echo "4. Enable auto-start on boot:"
echo "   sudo systemctl enable jetson-client"
echo ""
echo "5. Check status:"
echo "   sudo systemctl status jetson-client"
echo ""
echo "6. View logs:"
echo "   sudo journalctl -u jetson-client -f"
echo "   or: sudo tail -f /var/log/jetson-client.log"
echo ""
echo "============================================"
