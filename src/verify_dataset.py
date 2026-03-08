import os

IMAGE_DIR = r"C:\Users\riyas\OneDrive\Documents\Traffic-Congestion\Data\train\images"
LABEL_DIR = r"C:\Users\riyas\OneDrive\Documents\Traffic-Congestion\Data\train\labels"


def verify_dataset():
    image_files = sorted(os.listdir(IMAGE_DIR))
    label_files = sorted(os.listdir(LABEL_DIR))

    image_names = set([os.path.splitext(f)[0] for f in image_files])
    label_names = set([os.path.splitext(f)[0] for f in label_files])

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    print(f"Total Images: {len(image_names)}")
    print(f"Total Labels: {len(label_names)}")

    if missing_labels:
        print("\nImages without labels:")
        print(missing_labels)
    else:
        print("\nAll images have matching labels.")

    if missing_images:
        print("\nLabels without images:")
        print(missing_images)
    else:
        print("All labels have matching images.")

if __name__ == "__main__":
    verify_dataset()
