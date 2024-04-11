import os
import csv
from pathlib import Path
import re

# The directory containing your files
directory = Path('experiments/individual_n_tests_outputs')

# The pattern to extract n, m, and b from filenames
pattern = r'accept_length_n(\d+)_m([\d.]+)-b(\d+).out'

# The CSV file where you want to save the data
csv_filename = directory / 'collected_output.csv'

# Initialize a list to hold all the rows
rows = []

# List all files in the directory
for filename in os.listdir(directory):
    # Check if the filename matches the pattern
    match = re.match(pattern, filename)
    if match:
        # Extract n, m, and b values
        n, m, b = match.groups()
        
        # Read the floating-point number from the file
        with open(os.path.join(directory, filename), 'r') as file:
            number = file.read().strip()
        
        # Append the extracted data to the rows list
        rows.append([n, m, b, number])

# Write the collected data into a CSV file
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['n', 'm', 'b', 'number'])
    # Write all rows
    writer.writerows(rows)

print('Data has been successfully collected into', csv_filename)