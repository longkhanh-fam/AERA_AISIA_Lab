import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms
from torchvision.models import resnext50_32x4d  # ResNeXt-50, 32x4d version
import torch
# Initialize the model for feature extraction
model = resnext50_32x4d(pretrained=True)
model.eval()
# Replace the final layer with an identity function
model.fc = torch.nn.Identity()
# Define the preprocessing pipeline

dataset_base_path = 'output/segmented'
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#Initialize the model for feature extraction
model = models.resnet50(pretrained=True)
model.eval()  # Set model to evaluation mode
model.fc = torch.nn.Identity()  # Replace the final layer with an identity function
# Feature extraction function
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()
def process_dataset(dataset_path):
    image_paths = []
    feature_vectors = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.png'):
                image_path = os.path.join(root, file)
                features = extract_features(image_path)
                #features = extract_features(image_path, learn)
                feature_vectors.append(features.flatten())
                image_paths.append(image_path)
                
    return np.array(feature_vectors), image_paths

def create_annoy_index(feature_vectors, n_trees=10):
    dimension = feature_vectors.shape[1]
    index = AnnoyIndex(dimension, 'euclidean')  # Using Euclidean distance
    for i, vector in enumerate(feature_vectors):
        index.add_item(i, vector)
    index.build(n_trees)  # n_trees is a parameter affecting the build time and the query time.
    return index

def recommend_similar_items_annoy(image_path, index, image_paths_dataset, top_k=5):
    query_features = extract_features(image_path).flatten()
    #query_features = extract_features(image_path, learn).flatten()
    nearest_ids = index.get_nns_by_vector(query_features, top_k + 1, include_distances=False)
    
    # Filter out the query image from the results
    similar_image_paths = [image_paths_dataset[i] for i in nearest_ids if image_paths_dataset[i] != image_path][:top_k]
    
    return similar_image_paths


def display_similar_images(similar_items):
    # Define the number of images per row
    images_per_row = len(similar_items)
    
    # Calculate the number of rows needed
    rows = math.ceil(len(similar_items) / images_per_row)
    
    # Set figure size dynamically based on the number of images
    plt.figure(figsize=(20, 4 * rows))
    
    for i, image_path in enumerate(similar_items):
        img = mpimg.imread(image_path)
        plt.subplot(rows, images_per_row, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Image {i+1}")

    plt.tight_layout()
    plt.show()
