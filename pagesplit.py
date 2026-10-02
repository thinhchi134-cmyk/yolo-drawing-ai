from pathlib import Path
import random
import shutil

# =========================
# CONFIG
# =========================

SOURCE_IMAGES = Path("final_dataset/image")
SOURCE_LABELS = Path("final_dataset/label")

OUTPUT = Path("dataset_fine_tune_all_final")

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

SEED = 42


# =========================
# CHECK
# =========================

print("Current directory:")
print(Path.cwd())

print("\nImage folder:")
print(SOURCE_IMAGES.resolve())

print("Exists:", SOURCE_IMAGES.exists())

print("\nLabel folder:")
print(SOURCE_LABELS.resolve())

print("Exists:", SOURCE_LABELS.exists())


# =========================
# FIND IMAGES
# =========================

images = []

for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"]:
    images.extend(SOURCE_IMAGES.glob(ext))

images = sorted(images)

print("\nImages found:", len(images))

if len(images) == 0:
    raise RuntimeError(
        "Không tìm thấy ảnh! Kiểm tra SOURCE_IMAGES."
    )


# =========================
# SHUFFLE
# =========================

random.seed(SEED)

random.shuffle(images)


# =========================
# SPLIT
# =========================

n = len(images)

n_train = int(n * TRAIN_RATIO)
n_val = int(n * VAL_RATIO)

train_images = images[:n_train]

val_images = images[
    n_train:n_train + n_val
]

test_images = images[
    n_train + n_val:
]


print("\nSplit:")
print("Train:", len(train_images))
print("Val:  ", len(val_images))
print("Test: ", len(test_images))


# =========================
# CREATE FOLDERS
# =========================

for split in ["train", "val", "test"]:

    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# =========================
# COPY FUNCTION
# =========================

def copy_dataset(images, split):

    for image_path in images:

        # Copy image
        destination_image = (
            OUTPUT
            / "images"
            / split
            / image_path.name
        )

        shutil.copy2(
            image_path,
            destination_image
        )


        # Find label
        label_path = (
            SOURCE_LABELS
            / f"{image_path.stem}.txt"
        )

        destination_label = (
            OUTPUT
            / "labels"
            / split
            / label_path.name
        )


        if label_path.exists():

            shutil.copy2(
                label_path,
                destination_label
            )

        else:

            print(
                f"WARNING: Missing label: "
                f"{image_path.name}"
            )


# =========================
# COPY DATA
# =========================

copy_dataset(train_images, "train")

copy_dataset(val_images, "val")

copy_dataset(test_images, "test")


print("\nDone!")

print("\nDataset location:")
print(OUTPUT.resolve())