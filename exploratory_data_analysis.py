# step2_eda_save.py
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==== SETTINGS ====
#CSV_PATH     = 'faces_cleaned.csv' if os.path.exists('faces_cleaned.csv') else 'faces.csv'
CSV_PATH     = 'faces.csv'
IMAGE_FOLDER = 'images/images'
PLOT_DIR     = 'eda_plots'

# Create directory for saving plots
os.makedirs(PLOT_DIR, exist_ok=True)

# ==== LOAD ANNOTATIONS ====
df = pd.read_csv(CSV_PATH)
df.drop_duplicates(subset=['image_name','x0','y0','x1','y1'], inplace=True)

# ---- 2.1 IMAGE & FACE COUNT ----
total_images = df['image_name'].nunique()
total_faces  = len(df)
print(f"🖼️ Unique images: {total_images}")
print(f"👤 Total face boxes: {total_faces}")

# ---- 2.2 FACE COUNT PER IMAGE ----
face_counts = df.groupby('image_name').size()
print("\n📊 Face count per image stats:")
print(face_counts.describe())

plt.figure(figsize=(8,4))
sns.histplot(face_counts, bins=20, kde=False)
plt.title("Faces per Image")
plt.xlabel("Number of Faces")
plt.ylabel("Number of Images")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'faces_per_image_distribution.png'))
plt.close()

# ---- 2.3 BOUNDING BOX ACCURACY ----
invalid = df[
    (df.x0 >= df.x1) | (df.y0 >= df.y1) |
    (df.x0 < 0)  | (df.y0 < 0)
]
print(f"\n🚫 Invalid bounding boxes: {len(invalid)}")

# ---- 2.4 LABEL CONSISTENCY (BOX SIZES) ----
df['box_width']  = df.x1 - df.x0
df['box_height'] = df.y1 - df.y0
print("\n📏 Bounding box size summary:")
print(df[['box_width','box_height']].describe())

plt.figure(figsize=(6,6))
sns.scatterplot(x='box_width', y='box_height', data=df, alpha=0.3)
plt.title("Bounding Box Width vs Height")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'bbox_width_vs_height.png'))
plt.close()

# ---- 2.5 RESIZE REQUIREMENTS ----
resize_needed = df[(df.box_width  != 224) | (df.box_height != 224)]
print(f"\n🔧 Boxes not 224×224: {len(resize_needed)} of {len(df)}")

# ---- 2.6 IMAGE RESOLUTION CHECK ----
resolutions = []
for img_name in df['image_name'].unique():
    img_path = os.path.join(IMAGE_FOLDER, img_name)
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append((w, h))

res_df = pd.DataFrame(resolutions, columns=['width','height'])
print("\n🖼️ Image resolution summary:")
print(res_df.describe())

plt.figure(figsize=(8,4))
sns.histplot(res_df['width'], label='Width', kde=True)
sns.histplot(res_df['height'], label='Height', kde=True)
plt.legend()
plt.title("Image Resolution Distribution")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, 'image_resolution_distribution.png'))
plt.close()

print(f"\n✅ EDA plots saved to '{PLOT_DIR}'")
