import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==== SETTINGS ====
IMAGE_FOLDER       = 'images/images'    # adjust if needed
CSV_PATH           = 'faces.csv'
OUTPUT_SHAPE       = (224, 224)
SAVE_CROPPED       = False              # set True to save per‐image crops
CROPPED_OUTPUT_DIR = 'preprocessed_faces'
AUG_PER_IMAGE      = 2                  # reduce for smaller output
COMPRESSED_FN      = 'faces_data_compressed.npz'

# ==== LOAD & CLEAN ANNOTATIONS ====
df = pd.read_csv(CSV_PATH)
df.drop_duplicates(subset=['image_name', 'x0', 'y0', 'x1', 'y1'], inplace=True)
df = df[(df.x1 > df.x0) & (df.y1 > df.y0)].reset_index(drop=True)

# ==== OPTIONAL: OUTPUT DIR FOR CROPPED IMAGES ====
if SAVE_CROPPED and not os.path.exists(CROPPED_OUTPUT_DIR):
    os.makedirs(CROPPED_OUTPUT_DIR)

# ==== IMAGE PREPROCESSING FUNCTION ====
def preprocess_image(row):
    path = os.path.join(IMAGE_FOLDER, row.image_name)
    if not os.path.exists(path):
        print(f"❌ Missing: {path}")
        return None

    img = cv2.imread(path)
    if img is None:
        print(f"❌ Corrupt: {path}")
        return None

    h, w = img.shape[:2]
    x0, y0, x1, y1 = map(int, [row.x0, row.y0, row.x1, row.y1])
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        print(f"❌ Bad box {path}: ({x0},{y0})–({x1},{y1})")
        return None

    face = img[y0:y1, x0:x1]
    try:
        face = cv2.resize(face, OUTPUT_SHAPE)
    except Exception as e:
        print(f"❌ Resize error {path}: {e}")
        return None

    face = face.astype('float32') / 255.0

    if SAVE_CROPPED:
        outp = os.path.join(CROPPED_OUTPUT_DIR, row.image_name)
        cv2.imwrite(outp, (face * 255).astype('uint8'))

    return face

# ==== STEP 1: PROCESS ALL IMAGES ====
faces = []
valid_rows = []

print("🔄 Processing images...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    f = preprocess_image(row)
    if f is not None:
        faces.append(f)
        valid_rows.append(row)

faces_array = np.stack(faces) if faces else np.empty((0, *OUTPUT_SHAPE, 3), dtype='float32')
valid_df     = pd.DataFrame(valid_rows)

print(f"✅ Preprocessed faces: {len(faces_array)}")

# ==== SAVE CLEANED ANNOTATIONS ====
valid_df.to_csv('faces_cleaned.csv', index=False)
print("➡️ Saved cleaned annotations to faces_cleaned.csv")

# ==== STEP 1b: DATA AUGMENTATION ====
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

augmented = []
for img in tqdm(faces_array):
    batch_iter = augmentor.flow(np.expand_dims(img, 0), batch_size=1)
    for _ in range(AUG_PER_IMAGE):
        augmented.append(batch_iter.next()[0])

augmented_array = np.stack(augmented) if augmented else np.empty((0, *OUTPUT_SHAPE, 3), dtype='float32')
print(f"✅ Augmented images: {len(augmented_array)}")

# ==== STEP 1c: COMPRESS & SAVE ARRAYS ====
# Downcast to half‐precision & save compressed .npz
faces_small = faces_array.astype(np.float16)
aug_small   = augmented_array.astype(np.float16)

np.savez_compressed(
    COMPRESSED_FN,
    clean=faces_small,
    augmented=aug_small
)
print(f"➡️ Saved compressed data to {COMPRESSED_FN}")
print(f"   clean ~ {faces_small.nbytes/1e6:.1f} MB, augmented ~ {aug_small.nbytes/1e6:.1f} MB")
