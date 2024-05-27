import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import pandas as pd

# Define the paths to your dataset
empty_dir = '../data/dataset/train/empty'
occupied_dir = '../data/dataset/train/occupied'

# Get the list of dataset in each category
empty_images = os.listdir(empty_dir)
occupied_images = os.listdir(occupied_dir)

# Print out the counts for each category
print(f"Number of empty dataset: {len(empty_images)}")
print(f"Number of occupied dataset: {len(occupied_images)}")

# Create a bar plot for class distribution
counts = [len(empty_images), len(occupied_images)]
categories = ['Empty', 'Occupied']

plt.figure(figsize=(8, 6))
sns.barplot(x=categories, y=counts, palette='viridis')
plt.title('Class Distribution of Parking Spot Images')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# Function to show sample dataset
def show_images(images, title, n=5):
    plt.figure(figsize=(15, 5))
    for i in range(n):
        img = Image.open(images[i])
        plt.subplot(1, n, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(title)
    plt.show()


# Display sample dataset from each category
show_images([os.path.join(empty_dir, img) for img in empty_images[:5]], 'Sample Empty Images')
show_images([os.path.join(occupied_dir, img) for img in occupied_images[:5]], 'Sample Occupied Images')


# Function to get image sizes in KB
def get_image_file_sizes(image_paths):
    sizes = []
    for img_path in image_paths:
        size = os.path.getsize(img_path) / 1024  # Convert bytes to kilobytes
        sizes.append(size)
    return sizes


# Get image file sizes for each category
empty_image_paths = [os.path.join(empty_dir, img) for img in empty_images]
occupied_image_paths = [os.path.join(occupied_dir, img) for img in occupied_images]

empty_file_sizes = get_image_file_sizes(empty_image_paths)
occupied_file_sizes = get_image_file_sizes(occupied_image_paths)

# Combine the data into a DataFrame for easier plotting
empty_df = pd.DataFrame({'FileSizeKB': empty_file_sizes, 'Status': 'Empty'})
occupied_df = pd.DataFrame({'FileSizeKB': occupied_file_sizes, 'Status': 'Occupied'})
sizes_df = pd.concat([empty_df, occupied_df])

# Calculate statistics for file sizes
file_size_stats = {
    'min': sizes_df['FileSizeKB'].min(),
    'mean': sizes_df['FileSizeKB'].mean(),
    'median': sizes_df['FileSizeKB'].median(),
    'max': sizes_df['FileSizeKB'].max()
}

# Plot percentage distribution with vertical lines for statistics
plt.figure(figsize=(14, 6))

# Calculate the total count of dataset
total_count = len(sizes_df)

# Plot the histogram
sns.histplot(data=sizes_df, x='FileSizeKB', hue='Status', element='step', stat='percent', common_norm=False, bins=30)

# Draw vertical lines for statistics
plt.axvline(file_size_stats['min'], color='red', linestyle='--', label=f"Min Size: {file_size_stats['min']:.2f} KB")
plt.axvline(file_size_stats['mean'], color='blue', linestyle='--', label=f"Mean Size: {file_size_stats['mean']:.2f} KB")
plt.axvline(file_size_stats['median'], color='green', linestyle='--',
            label=f"Median Size: {file_size_stats['median']:.2f} KB")
plt.axvline(file_size_stats['max'], color='purple', linestyle='--', label=f"Max Size: {file_size_stats['max']:.2f} KB")

# Add legend and labels
plt.legend()
plt.title('Percentage Distribution of Image File Sizes')
plt.xlabel('File Size (KB)')
plt.ylabel('Percentage')

# Adjust layout for better fit
plt.tight_layout()
plt.show()

# Define size categories
size_categories = [1.8, 2.0, 2.2, 2.4, 2.8]
size_labels = [f'{size} KB' for size in size_categories]

# Calculate the count of dataset within each size category
size_counts = [(sizes_df['FileSizeKB'] >= size).sum() for size in size_categories]

# Calculate the percentage of dataset within each size category
size_percentages = [(count / total_count) * 100 for count in size_counts]

# Plot the pie chart
plt.figure(figsize=(8, 8))
plt.pie(size_percentages, labels=size_labels, autopct='%1.1f%%', startangle=140,
        colors=sns.color_palette('viridis', len(size_categories)))
plt.title('Percentage Distribution of Image File Sizes in Specific Ranges')
plt.show()
