#!/bin/bash

# Specify the path to the folder containing the files
folder_path="stats"

# Check if the folder exists
if [ -d "$folder_path" ]; then
    # Change to the folder directory
    cd "$folder_path"
    
    # Use a loop to remove all files in the folder
    for file in *; do
        if [ -f "$file" ]; then
            rm "$file"
            # echo "Removed: $file"
        fi
    done

    echo "All files in $folder_path have been removed."
else
    echo "Folder $folder_path does not exist."
fi
