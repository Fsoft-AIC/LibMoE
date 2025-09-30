import os

def find_files_in_size_range(directory, min_size=30, max_size=45):
    # List to store file paths that meet the size condition
    matching_files = []

    # Walk through the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)

            # Check if the file size is between the given range
            if min_size < file_size < max_size:
                matching_files.append(file_path)

    return matching_files

def write_files_to_output_file(matching_files, output_file):
    with open(output_file, 'w') as f:
        for file in matching_files:
            f.write(file + '\n')

# Usage
directory = "/cm/archive/anonymous/checkpoints/logs"
output_file = "/cm/shared/anonymous_H102/toolkitmoe/evaluate/analysis/matching_files.txt"  # You can specify your desired output file name
matching_files = find_files_in_size_range(directory)

# Write the result to a file
write_files_to_output_file(matching_files, output_file)

print(f"Matching files have been written to {output_file}")
