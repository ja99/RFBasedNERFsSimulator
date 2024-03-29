import os
import shutil

def delete_folders_without_file(parent_folder, filename):
    # Walk through all the directories in the parent folder
    count = 0
    for dirpath, dirnames, files in os.walk(parent_folder, topdown=False):
        # Check if the directory is not the parent folder itself
        if dirpath != parent_folder:
            # Check if the file is not in the directory
            if filename not in files:
                try:
                    # Attempt to delete the folder
                    shutil.rmtree(dirpath)
                    print(f"Deleted folder: {dirpath}")
                    count += 1
                except OSError as e:
                    print(f"Error: {dirpath} : {e.strerror}")
    print(f"Deleted {count} folders")
# Specify the parent folder and the required file
parent_folder = 'beamforming_data'
filename = 'sensor_samples.npy'

# Call the function
delete_folders_without_file(parent_folder, filename)






