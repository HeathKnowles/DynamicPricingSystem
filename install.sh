#!/bin/bash

# Set up a virtual environment
echo "Creating virtual environment..."
python3 -m venv myenv

# Activate the virtual environment
echo "Activating virtual environment..."
source myenv/bin/activate

# Upgrade pip to ensure we're using the latest version
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing required packages..."
pip install -r requirements.txt

# Check if all dependencies installed correctly
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Exiting."
    exit 1
fi

echo "All dependencies installed successfully."

# Make the Bash script executable
chmod +x install.sh

# Run the Python files through the Bash script
echo "Running Python scripts..."
install.sh

# Deactivate the virtual environment
deactivate

echo "Installation and script execution completed successfully."
