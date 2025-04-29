import os
import pickle


def check_pickle_file(pickle_file):
    if not os.path.exists(pickle_file):
        print(f"File {pickle_file} does not exist.")
        return False

    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
            print(f"Successfully read pickle file: {pickle_file}")
    except Exception as e:
        print(f"Error reading pickle file: {e}")
        return False


check_pickle_file("data/flickr30k_clip_embeddings.pkl")
