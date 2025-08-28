import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== SETTINGS ====
IMAGE_FOLDER = 'images\images'               # Folder containing all face images
CSV_PATH = 'faces.csv'                # Annotation file
OUTPUT_SHAPE = (224, 224)             # Resize target
SAVE_CROPPED = False                  # Change to True to save preprocessed images
CROPPED_OUTPUT_DIR = 'preprocessed_faces'
AUG_PER_IMAGE      = 2                  # reduce for smaller output
COMPRESSED_FN      = 'faces_data_compressed.npz'

# ==== LOAD CSV ====
df = pd.read_csv(CSV_PATH)

# ==== REMOVE DUPLICATES & INVALID ENTRIES ====
df.drop_duplicates(subset=['image_name'], inplace=True)
df = df[df['x1'] > df['x0']]
df = df[df['y1'] > df['y0']]
df.reset_index(drop=True, inplace=True)

# ==== OPTIONAL: CREATE OUTPUT DIR ====
if SAVE_CROPPED and not os.path.exists(CROPPED_OUTPUT_DIR):
    os.makedirs(CROPPED_OUTPUT_DIR)

# ==== FUNCTION: PREPROCESS IMAGE ====
def preprocess_image(row):
    path = os.path.join(IMAGE_FOLDER, row['image_name'])
    
    if not os.path.exists(path):
        print(f"❌ Missing file: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"❌ Corrupt image: {path}")
        return None

    h, w = img.shape[:2]
    x0, y0, x1, y1 = map(int, [row['x0'], row['y0'], row['x1'], row['y1']])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    
    if x1 <= x0 or y1 <= y0:
        print(f"❌ Invalid crop for {path}: ({x0},{y0}) to ({x1},{y1})")
        return None

    # Crop and resize
    face = img[y0:y1, x0:x1]
    try:
        face = cv2.resize(face, OUTPUT_SHAPE)
    except Exception as e:
        print(f"❌ Resize failed for {path}: {e}")
        return None

    # Normalize
    face = face / 255.0

    # Optional save
    if SAVE_CROPPED:
        filename = os.path.join(CROPPED_OUTPUT_DIR, row['image_name'])
        cv2.imwrite(filename, (face * 255).astype(np.uint8))

    return face

# ==== APPLY PREPROCESSING ====
faces = []
valid_rows = []

print("🔄 Processing images...")
for i in tqdm(range(len(df))):
    row = df.iloc[i]
    face = preprocess_image(row)
    if face is not None:
        faces.append(face)
        valid_rows.append(row)

# Convert to arrays
faces_array = np.array(faces)
valid_df = pd.DataFrame(valid_rows)

print(f"\n✅ Done! Preprocessed {len(faces_array)} valid face images.")

# ==== DATA AUGMENTATION ====
print("🎨 Applying data augmentation...")
augmentor = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Example: Generate 5 augmented images per face
augmented_faces = []
for face in tqdm(faces_array):
    face = np.expand_dims(face, axis=0)
    aug_iter = augmentor.flow(face, batch_size=1)
    for _ in range(5):
        augmented_faces.append(next(aug_iter)[0])

augmented_faces_array = np.array(augmented_faces)
print(f"✅ Augmented dataset size: {augmented_faces_array.shape[0]}")

# ==== OPTIONAL: SAVE ARRAYS ====
# np.save('faces_clean.npy', faces_array)
# np.save('faces_augmented.npy', augmented_faces_array)

valid_df.to_csv('faces_cleaned.csv', index=False)  
print("➡️ Saved cleaned annotations to faces_cleaned.csv")

# ==== STEP 1c: COMPRESS & SAVE ARRAYS ====
# Downcast to half‐precision & save compressed .npz
faces_small = faces_array.astype(np.float16)
aug_small   = augmented_faces_array.astype(np.float16)

np.savez_compressed(
    COMPRESSED_FN,
    clean=faces_small,
    augmented=aug_small
)
print(f"➡️ Saved compressed data to {COMPRESSED_FN}")
print(f"   clean ~ {faces_small.nbytes/1e6:.1f} MB, augmented ~ {aug_small.nbytes/1e6:.1f} MB")