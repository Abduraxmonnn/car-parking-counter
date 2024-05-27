import os
import matplotlib.pyplot as plt


def count_images(directory):
    """
    Count the number of images in each category within the given directory.

    Args:
        directory (str): Path to the directory containing 'empty' and 'occupied' subdirectories.

    Returns:
        dict: A dictionary with the counts of images in 'empty' and 'occupied' categories.
    """
    categories = ['empty', 'occupied']
    counts = {}

    for category in categories:
        category_dir = os.path.join(directory, category)
        counts[category] = len(os.listdir(category_dir)) if os.path.exists(category_dir) else 0

    return counts


def plot_grouped_bar_chart(train_counts, validation_counts):
    """
    Plot a grouped bar chart to compare the number of images in 'train' and 'validation' directories.

    Args:
        train_counts (dict): Dictionary with counts of images in 'empty' and 'occupied' categories for training data.
        validation_counts (dict): Dictionary with counts of images in 'empty' and 'occupied' categories for validation data.
    """
    categories = ['Train', 'Validation']

    empty_values = [train_counts['empty'], validation_counts['empty']]
    occupied_values = [train_counts['occupied'], validation_counts['occupied']]

    x = range(len(categories))

    plt.figure(figsize=(10, 6))

    bar_width = 0.35
    plt.bar(x, empty_values, width=bar_width, label='Empty', alpha=0.7, color='b')
    plt.bar([p + bar_width for p in x], occupied_values, width=bar_width, label='Occupied', alpha=0.7, color='g')

    plt.xlabel('Dataset')
    plt.ylabel('Number of Images')
    plt.title('Comparison of Empty and Occupied Image Counts')
    plt.xticks([p + bar_width / 2 for p in x], categories)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()

    # Save the plot
    output_dir = '../data/results/'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'grouped_bar_chart.png'))

    # Show the plot
    plt.show()


def main():
    # Define the directories for training and validation data
    train_dir = '../data/dataset/train'
    validation_dir = '../data/dataset/validation'

    # Count the number of images in each category
    train_counts = count_images(train_dir)
    validation_counts = count_images(validation_dir)

    # Print the counts for verification
    print('Training counts:', train_counts)
    print('Validation counts:', validation_counts)

    # Plot the grouped bar chart
    plot_grouped_bar_chart(train_counts, validation_counts)


if __name__ == "__main__":
    main()
