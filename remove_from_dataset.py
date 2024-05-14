import os
import csv
from PIL import Image

input_file = "small_training_dataset_with_prompts.csv"  # Path to your input CSV file
output_file = "training_data.csv"  # Path to the output CSV file

def right_size(path):
    img = Image.open(path)
    if img.size != (1536, 662):
        return False
    else:
        return True

# Open input and output CSV files
with open(input_file, 'r', newline='') as csvfile, open(output_file, 'w', newline='') as cleaned_csvfile:
    reader = csv.reader(csvfile)
    writer = csv.writer(cleaned_csvfile)

    # Iterate through each row in the input CSV file
    for row in reader:
        # Extract the file path from the row
        file_path = row[0]

        # Check if the file exists
        if right_size(file_path):
            # Write the row to the output CSV file if the file exists
            writer.writerow(row)
        else:
            # If the file does not exist, skip this row
            print(f"File not found: {file_path}")

print("Cleaning complete.")
