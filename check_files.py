import glob
import os

# Check for jpg files
files = glob.glob(r'C:\urbanfit\datasets\train\*\*.jpg')
print(f"Found {len(files)} images.")

if len(files) == 0:
    print("No images found! Check if they are actually .jpg files.")
else:
    print("Images found! First one is at:", files[0])