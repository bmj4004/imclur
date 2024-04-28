import time
import os, os.path
import random
import cv2
import keras
import matplotlib
import matplotlib.pyplot as plt
import glob2
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import pandas as pd
import numpy as np

# directory where images are stored
DIR = './disaster_images'

def dataset_stats():
    disaster_characters = ['D', 'E', 'F', 'L', 'W']
    
    # dictionary where we will store the stats
    stats = []
    
    for disaster in disaster_characters:
        # get a list of subdirectories that start with this character
        directory_list = sorted(glob2.glob("{}/[{}]*".format(DIR, disaster)))
        
        for sub_directory in directory_list:
            file_names = [file for file in os.listdir(sub_directory)]
            file_count = len(file_names)
            sub_directory_name = os.path.basename(sub_directory)
            stats.append({ "Code": sub_directory_name[:sub_directory_name.find('-')],
                            "Image count": file_count, 
                           "Folder name": os.path.basename(sub_directory),
                            "File names": file_names})
    
    
    df = pd.DataFrame(stats)
    
    return df


# Function returns an array of images whoose filenames start with a given set of characters
# after resizing them to 224 x 224
def load_images(df, codes):
    # Define empty arrays where we will store our images and labels
    images = []
    labels = []
    
    for code in codes:
        # get the folder name for this code
        folder_name = df.loc[code]["Folder name"]
        
        for file in df.loc[code]["File names"]:                 
            # build file path
            file_path = os.path.join(DIR, folder_name, file)
        
            # Read the image
            image = cv2.imread(file_path)

            # Resize it to 224 x 224
            image = cv2.resize(image, (224,224))

            # Convert it from BGR to RGB so we can plot them later (because openCV reads images as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Now we add it to our array
            images.append(image)
            labels.append(code)

    return images, labels


def normalise_images(images, labels):

    # Convert to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    # Normalise the images
    images /= 255
    
    return images, labels
    

def shuffle_data(images, labels):
    # Set aside the testing data. We won't touch these until the very end.
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=728)
    
    return X_train, y_train, X_test, y_test


def covnet_transform(covnet_model, raw_images):
    # Pass our training data through the network
    pred = covnet_model.predict(raw_images)

    # Flatten the array
    flat = pred.reshape(raw_images.shape[0], -1)
    
    return flat


def create_fit_PCA(data, n_components=None):
    
    p = PCA(n_components=n_components, random_state=728)
    p.fit(data)
    
    return p


def create_train_kmeans(data, number_of_clusters):
    # n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    # especially when the data size gets much bigger. #perfMatters
    
    k = KMeans(n_clusters=number_of_clusters, random_state=728)

    # Let's do some timings to see how long it takes to train.
    start = time.time()

    # Train it up
    k.fit(data)

    # Stop the timing 
    end = time.time()

    # And see how long that took
    print("Training took {} seconds".format(end-start))
    
    return k


def create_train_gmm(data, number_of_clusters):
    g = GaussianMixture(n_components=number_of_clusters, covariance_type="full", random_state=728)
    
    start=time.time()
    g.fit(data)
    end=time.time()
    
    print("Training took {} seconds".format(end-start))
    
    return g


def cluster_label_count(clusters, labels):
    
    count = {}
    
    # Get unique clusters and labels
    unique_clusters = list(set(clusters))
    unique_labels = list(set(labels))
    
    # Create counter for each cluster/label combination and set it to 0
    for cluster in unique_clusters:
        count[cluster] = {}
        
        for label in unique_labels:
            count[cluster][label] = 0
    
    # Let's count
    for i in range(len(clusters)):
        count[clusters[i]][labels[i]] +=1
    
    cluster_df = pd.DataFrame(count)
    
    return cluster_df


from sklearn.metrics import accuracy_score, f1_score

def print_scores(true, pred):
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")
    return "\n\tF1 Score: {0:0.8f}   |   Accuracy: {0:0.8f}".format(f1,acc)


if __name__ == "__main__":
    dataset = dataset_stats().set_index("Code")
    print(dataset[["Folder name", "File names", "Image count"]])

    codes = ['D', 'E', 'F', 'L', 'W']
    images, labels = load_images(dataset, codes)
    images, labels = normalise_images(images, labels)
    X_train, y_train, x_test, y_test = shuffle_data(images, labels)
    
    vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3))
    vgg19_model = keras.applications.vgg19.VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3))
    resnet50_model = keras.applications.resnet50.ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))

    vgg16_output = covnet_transform(vgg16_model, X_train)
    print("VGG16 flattened output has {} features".format(vgg16_output.shape[1]))

    vgg19_output = covnet_transform(vgg19_model, X_train)
    print("VGG19 flattened output has {} features".format(vgg19_output.shape[1]))

    resnet50_output = covnet_transform(resnet50_model, X_train)
    print("ResNet50 flattened output has {} features".format(resnet50_output.shape[1]))
    
    # Create PCA instances for each covnet output
    vgg16_pca = create_fit_PCA(vgg16_output)
    vgg19_pca = create_fit_PCA(vgg19_output)
    resnet50_pca = create_fit_PCA(resnet50_output)
    
    # PCA transformations of covnet outputs
    vgg16_output_pca = vgg16_pca.transform(vgg16_output)
    vgg19_output_pca = vgg19_pca.transform(vgg19_output)
    resnet50_output_pca = resnet50_pca.transform(resnet50_output)
    
    print("KMeans (PCA): \n")
    print("VGG16")
    K_vgg16_pca = create_train_kmeans(vgg16_output_pca, len(codes))
    print("\nVGG19")
    K_vgg19_pca = create_train_kmeans(vgg19_output_pca, len(codes))
    print("\nResNet50")
    K_resnet50_pca = create_train_kmeans(resnet50_output_pca, len(codes))
    
    print("GMM (PCA): \n")
    print("VGG16")
    G_vgg16_pca = create_train_gmm(vgg16_output_pca, len(codes))
    print("\nVGG19")
    G_vgg19_pca = create_train_gmm(vgg19_output_pca, len(codes))
    print("\nResNet50")
    G_resnet50_pca = create_train_gmm(resnet50_output_pca, len(codes))
    
    # Let's also create models for the covnet outputs without PCA for comparison
    print("KMeans: \n")
    print("VGG16:")
    K_vgg16 = create_train_kmeans(vgg16_output, len(codes))
    print("\nVGG19:")
    K_vgg19 = create_train_kmeans(vgg19_output, len(codes))
    print("\nResNet50:")
    K_resnet50 = create_train_kmeans(resnet50_output, len(codes))
    
    # Now we get the custer model predictions

    # KMeans with PCA outputs
    k_vgg16_pred_pca = K_vgg16_pca.predict(vgg16_output_pca)
    k_vgg19_pred_pca = K_vgg19_pca.predict(vgg19_output_pca)
    k_resnet50_pred_pca = K_resnet50_pca.predict(resnet50_output_pca)
    # KMeans with CovNet outputs
    k_vgg16_pred = K_vgg16.predict(vgg16_output)
    k_vgg19_pred = K_vgg19.predict(vgg19_output)
    k_resnet50_pred = K_resnet50.predict(resnet50_output)
    # Gaussian Mixture with PCA outputs
    g_resnet50_pred_pca = G_resnet50_pca.predict(resnet50_output_pca)
    g_vgg16_pred_pca = G_vgg16_pca.predict(vgg16_output_pca)
    g_vgg19_pred_pca = G_vgg19_pca.predict(vgg19_output_pca)
    
    # Cluster counting for VGG16 Means
    vgg16_cluster_count = cluster_label_count(k_vgg16_pred, y_train)
    vgg16_cluster_count_pca = cluster_label_count(k_vgg16_pred_pca, y_train)
    # VGG19 KMeans
    vgg19_cluster_count = cluster_label_count(k_vgg19_pred, y_train)
    vgg19_cluster_count_pca = cluster_label_count(k_vgg19_pred_pca, y_train)
    # ResNet50 KMeans
    resnet_cluster_count = cluster_label_count(k_resnet50_pred, y_train)
    resnet_cluster_count_pca = cluster_label_count(k_resnet50_pred_pca, y_train)
    # GMM
    g_vgg16_cluster_count_pca = cluster_label_count(g_vgg16_pred_pca, y_train)
    g_vgg19_cluster_count_pca = cluster_label_count(g_vgg19_pred_pca, y_train)
    g_resnet50_cluster_count_pca = cluster_label_count(g_resnet50_pred_pca, y_train)
    
    print("KMeans VGG16: ")
    print(vgg16_cluster_count)

    print("KMeans VGG16 (PCA): ")
    print(vgg16_cluster_count_pca)
    
    print("GMM VGG16: ")
    print(g_vgg16_cluster_count_pca)
    
    print("KMeans VGG19: ")
    print(vgg19_cluster_count)
    
    print("KMeans VGG19 (PCA): ")
    print(vgg19_cluster_count_pca)
    
    print("GMM VGG19 (PCA): ")
    print(g_vgg19_cluster_count_pca)
    
    print("KMeans Resnet50: ")
    print(resnet_cluster_count)
    
    print("Kmeans Resnet50 (PCA): ")
    print(resnet_cluster_count_pca)
    
    print("GMM Resnet50 (PCA): ")
    print(g_resnet50_cluster_count_pca)
    
    vgg16_pred_codes = [codes[x] for x in k_vgg16_pred]
    vgg16_pred_codes_pca = [codes[x] for x in k_vgg16_pred_pca]
    vgg19_pred_codes = [codes[x] for x in k_vgg19_pred]
    vgg19_pred_codes_pca = [codes[x] for x in k_vgg19_pred_pca]
    g_vgg19_pred_codes_pca = [codes[x] for x in g_vgg19_pred_pca]
    
    print("KMeans VGG16:", print_scores(y_train, vgg16_pred_codes))
    print("KMeans VGG16 (PCA)", print_scores(y_train, vgg16_pred_codes_pca))
    print("\nKMeans VGG19: ", print_scores(y_train, vgg19_pred_codes))
    print("KMeans VGG19 (PCA): ", print_scores(y_train, vgg19_pred_codes_pca))
    print("GMM VGG19 (PCA)", print_scores(y_train, g_vgg19_pred_codes_pca))