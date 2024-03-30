import csv
import os
from path import *

def fill_missing_values(input_csv_path, output_csv_path):
    with open(input_csv_path, 'r') as input_file, open(output_csv_path, 'w', newline='') as output_file:
        reader = csv.reader(input_file)
        writer = csv.writer(output_file)

        previous_values = [None, None]  # Initialize previous values for the second and third columns

        for row in reader:
            filled_row = row[:]  # Create a copy of the original row
            for i in range(1, 3):  # Iterate over the second and third columns
                if filled_row[i] == '-':  # If the value is missing
                    filled_row[i] = previous_values[i - 1]  # Fill with the previous value
                else:
                    previous_values[i - 1] = filled_row[i]  # Update the previous value
            writer.writerow(filled_row)

# Example usage:
input_csv_path = os.path.join(BB_COORDINATES_DIR, 'moving_circle_14.csv')
output_csv_path = os.path.join(PREPROCESSED_BB_COORDINATES_DIR, 'moving_circle_14.csv')
fill_missing_values(input_csv_path, output_csv_path)
