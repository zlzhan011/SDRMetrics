import  os
import os
import shutil

def distribute_files_to_subfolders(parent_folder, num_subfolders=10):
    """
    Distribute files from a parent folder into a specified number of subfolders.
    If the subfolders don't exist, they will be created.

    :param parent_folder: The path to the parent folder containing the files.
    :param num_subfolders: The number of subfolders to distribute the files into.
    """
    # Create a list of subfolder names
    subfolder_names = [f"Subfolder_{i+1}" for i in range(num_subfolders)]

    # Create subfolders if they don't exist
    for subfolder in subfolder_names:
        path = os.path.join(parent_folder, subfolder)
        if not os.path.exists(path):
            os.makedirs(path)

    # List all files in the parent folder
    files = [f for f in os.listdir(parent_folder) if os.path.isfile(os.path.join(parent_folder, f))]

    # Distribute files into subfolders
    for index, file in enumerate(files):
        destination_folder = os.path.join(parent_folder, subfolder_names[index % num_subfolders])
        shutil.move(os.path.join(parent_folder, file), destination_folder)

# Example usage
# parent_folder_path = 'path/to/your/parent/folder'
# distribute_files_to_subfolders(parent_folder_path)

if __name__ == '__main__':
    parent_folder = '/data/cs_lzhan011/data/cs_lzhan011/vulnerability/data-package/models/Devign_ReVeal/code/data/ReVeal/raw_code'
    distribute_files_to_subfolders(parent_folder, num_subfolders=20)





