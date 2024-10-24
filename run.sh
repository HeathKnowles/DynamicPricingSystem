#!/bin/bash

# Ensure the script is run in the correct environment (if using virtualenv, activate it)
# source /path/to/virtualenv/bin/activate

# File for script 1 output
OUTPUT_FILE_1="output_script1.txt"
echo "Starting Python Script 1: commerce.py"
python3 script1.py > "$OUTPUT_FILE_1" 2>&1

# Check if the first script ran successfully
if [ $? -ne 0 ]; then
    echo "Script 1 failed. Exiting."
    exit 1
fi
echo "Output of script1.py saved to $OUTPUT_FILE_1"

# File for script 2 output
OUTPUT_FILE_2="output_script2.txt"
echo "Starting Python Script 2: demand_forecasting.py"
python3 script2.py > "$OUTPUT_FILE_2" 2>&1

# Check if the second script ran successfully
if [ $? -ne 0 ]; then
    echo "Script 2 failed. Exiting."
    exit 1
fi
echo "Output of script2.py saved to $OUTPUT_FILE_2"

# File for script 3 output
OUTPUT_FILE_3="output_script3.txt"
echo "Starting Python Script 3: sentiment_analysis.py"
python3 script3.py > "$OUTPUT_FILE_3" 2>&1

# Check if the third script ran successfully
if [ $? -ne 0 ]; then
    echo "Script 3 failed. Exiting."
    exit 1
fi
echo "Output of script3.py saved to $OUTPUT_FILE_3"

echo "All scripts ran successfully and output saved to respective files."
