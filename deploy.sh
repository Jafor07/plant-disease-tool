#!/bin/bash
# Quick Deployment Setup Script for Plant Disease Tool
# This script automates the deployment setup process

set -e

echo "🚀 Plant Disease Tool - Quick Deployment Setup"
echo "=================================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed."
    echo "📖 Install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed."
    echo "📖 Install from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker is installed ($(docker --version))"
echo "✅ Docker Compose is installed ($(docker-compose --version))"
echo ""

# Detect OS
OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then
    PLATFORM="Linux"
elif [[ "$OS" == "Darwin" ]]; then
    PLATFORM="macOS"
else
    PLATFORM="Windows"
fi

echo "📊 Detected OS: $PLATFORM"
echo ""

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p raw_images processed_images masks labels metadata
echo "✅ Directories created"
echo ""

# Check if .streamlit/config.toml exists
if [ -f ".streamlit/config.toml" ]; then
    echo "✅ Config file exists (.streamlit/config.toml)"
else
    echo "⚠️  Creating default config..."
    mkdir -p .streamlit
fi

# Check if Dockerfile exists
if [ -f "Dockerfile" ]; then
    echo "✅ Dockerfile exists"
else
    echo "⚠️  Dockerfile not found. Please create it from DEPLOYMENT.md"
fi

# Check if docker-compose.yml exists
if [ -f "docker-compose.yml" ]; then
    echo "✅ docker-compose.yml exists"
else
    echo "⚠️  docker-compose.yml not found. Please create it from DEPLOYMENT.md"
fi

echo ""
echo "=================================================="
echo "🎯 Ready to Deploy!"
echo "=================================================="
echo ""
echo "Option 1: Build Docker image"
echo "  $ docker build -t plant-disease-tool:latest ."
echo ""
echo "Option 2: Use Docker Compose (recommended)"
echo "  $ docker-compose up -d"
echo ""
echo "Option 3: Run locally (no Docker)"
echo "  $ streamlit run app.py"
echo ""
echo "=================================================="
echo "📖 Full deployment guide: See DEPLOYMENT.md"
echo "=================================================="
