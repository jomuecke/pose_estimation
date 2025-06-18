import os

def rename_files_and_subfolders(base_directory):
    # Walk through each folder inside the base directory
    for folder_name in os.listdir(base_directory):
        folder_path = os.path.join(base_directory, folder_name)
        
        # Ensure it's a directory (e.g., '110' in the structure you mentioned)
        if os.path.isdir(folder_path):
            # Now go through each of the identifier folders inside it (e.g., 'video-18-26-8_4')
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                
                # Ensure it's a directory (e.g., 'video-18-26-8_4' or 'video-18-26-8_5')
                if os.path.isdir(subfolder_path):
                    # Rename the subfolder
                    # Extract the ID from the subfolder name (assuming it looks like 'video-18-26-8_4')
                    id = subfolder.split('_')[1]
                    new_subfolder_name = f"images_R16_20250314_M{folder_name}_{id}"
                    new_subfolder_path = os.path.join(folder_path, new_subfolder_name)
                    
                    # Rename the subfolder
                    os.rename(subfolder_path, new_subfolder_path)
                    print(f"Renamed subfolder: {subfolder_path} -> {new_subfolder_path}")
                    
                    # Rename each image inside the subfolder
                    for image_name in os.listdir(new_subfolder_path):
                        # Process only PNG images
                        if image_name.endswith('.png'):
                            image_path = os.path.join(new_subfolder_path, image_name)
                            
                            # Extract frame number from the image name
                            parts = image_name.split('_')
                            id_and_frame = parts[1]  # The part with the id and frame number
                            split_id_frame = id_and_frame.split('-')
                            
                            # Assuming the frame number is before the extension
                            ID = split_id_frame[0]
                            frame_and_file = split_id_frame[1]  # Frame number without extension
                            split2 = frame_and_file.split('.')
                            frame_number = split2[0]
                            # Generate the new image name
                            new_image_name = f"images_R16_20250314_M{folder_name}_{ID}_F{frame_number}.png"
                            new_image_path = os.path.join(new_subfolder_path, new_image_name)
                            
                            # Rename the image
                            os.rename(image_path, new_image_path)
                            print(f"Renamed image: {image_path} -> {new_image_path}")

# Specify the base directory (update this with the path to your "extract_R17" folder)
base_directory = '/Users/jonasmucke/Desktop/R16'
rename_files_and_subfolders(base_directory)
