from retreive_images import *
def get_image_category(image_path):
    # Extracts the category name from the image path
    return image_path.split('/')[-2]  # Adjust this based on your actual path structure
import os
from collections import defaultdict

# Preparing the dataset: Map each category to its images
category_to_images = defaultdict(list)
for root, dirs, files in os.walk(dataset_base_path):
    for file in files:
        if file.endswith('.jpg') or file.endswith('.png'):  # Check your dataset's image format
            image_path = os.path.join(root, file)
            category = get_image_category(image_path)
            category_to_images[category].append(image_path)

import time  # Import the time module

def evaluate_single_image(query_image_path, category_to_images, annoy_index, image_paths_dataset):
    start_time = time.time()  # Record the start time
    
    query_category = get_image_category(query_image_path)
    relevant_images = category_to_images[query_category]
    
    # Exclude the query image from its own relevant set
    relevant_images = [img for img in relevant_images if img != query_image_path]
    
    recommended_images = recommend_similar_items_annoy(query_image_path, annoy_index, image_paths_dataset, top_k=4)
    query_time = time.time() - start_time  # Calculate the query time
    
    recommended_categories = [get_image_category(img) for img in recommended_images]
    
    # Calculate precision and recall
    true_positives = sum([1 for cat in recommended_categories if cat == query_category])
    precision = true_positives / len(recommended_images)
    recall = true_positives / len(relevant_images)
    
    return precision, recall, query_time  # Return the query time along with precision and recall


def evaluate_dataset(category_to_images, annoy_index, image_paths_dataset):
    precisions = []
    recalls = []
    query_times = []  # List to store all query times
    
    for category, images in category_to_images.items():
        for image_path in images:
            precision, recall, query_time = evaluate_single_image(image_path, category_to_images, annoy_index, image_paths_dataset)
            precisions.append(precision)
            recalls.append(recall)
            query_times.append(query_time)  # Store each query time
    
    # Compute mean precision, recall, and query time
    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    mean_query_time = sum(query_times) / len(query_times)  # Calculate the mean query time
    
    return mean_precision, mean_recall, mean_query_time  # Return the average query time along with precision and recall

# Perform evaluation
mean_precision, mean_recall, mean_query_time = evaluate_dataset(category_to_images, annoy_index, image_paths_dataset)
print(f"Mean Precision: {mean_precision}")
print(f"Mean Recall: {mean_recall}")
print(f"Average Query Time: {mean_query_time} seconds")

