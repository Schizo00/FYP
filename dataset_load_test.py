import tensorflow as tf
import os

def _parse_function(file_path):
  """Parses a file path into features (mesh and embeddings)."""

  # Extract folder name (id)
  folder_name = tf.strings.split(file_path, os.sep)[-2]

  # Assuming the parent folder of the file contains the mesh data
  mesh_path = tf.strings.join([os.path.dirname(file_path), "mesh.npy"], separator=os.sep)

  # Assuming embedding files are in a separate folder named "Embeddings"
  embeddings_path = tf.strings.join([os.path.dirname(file_path), "Embeddings"], separator=os.sep)

  # Load mesh data (replace with your actual loading logic)
  mesh = tf.io.read_file(mesh_path)
  mesh = tf.io.decode_npy(mesh)  # Assuming mesh is stored as a NumPy array

  # Load embedding data (replace with your actual loading logic)
  embeddings_paths = tf.strings.split(tf.io.listdir(embeddings_path), sep=".mp3.npy")
  embeddings = tf.strings.to_number(embeddings_paths, out_type=tf.float32)  # Assuming embeddings are NumPy arrays

  # Create features dictionary
  features = {
      "folder_name": folder_name,
      "mesh": mesh,
      "embeddings": embeddings,
  }

  return features

# Define dataset creation function
def create_dataset(data_dir):
  """Creates a TFDS dataset from the given directory."""

  # List all file paths within the data directory
  file_paths = tf.io.gfile.glob(data_dir + os.sep + "*/*")

  # Create dataset from file paths
  dataset = tf.data.Dataset.from_tensor_slices(file_paths)

  # Apply parsing function to each file path
  dataset = dataset.map(_parse_function)

  return dataset

# Example usage
data_dir = "C:/Users/nawee/Desktop/Dataset/"  
dataset = create_dataset(data_dir)

# Explore the dataset (optional)
for element in dataset.take(1):
  print(element)
