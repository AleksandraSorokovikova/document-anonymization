import os
import zipfile
import gdown
import shutil

# ==== CONFIGURATION ====
FILE_ID = "1Ht6PHCGK8U1U3hINlDo2AmQGtPVYC69U"
FOLDER_NAME = "supplementary_material"
ZIP_NAME = f"{FOLDER_NAME}.zip"
DOWNLOAD_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
print(f"🔗 Download URL: {DOWNLOAD_URL}")


if __name__ == "__main__":
    # ==== DOWNLOAD ====
    print("🔽 Downloading archive...")
    gdown.download(DOWNLOAD_URL, ZIP_NAME, quiet=False)

    if not os.path.exists(ZIP_NAME):
        print("❌ Failed to download the archive.")
        print(f"Please download the archive manually from {DOWNLOAD_URL} and place folders `weights/` and `data/` in the current directory.")
        exit(1)

    # ==== UNZIP ====
    print("📦 Unzipping archive...")
    with zipfile.ZipFile(ZIP_NAME, "r") as zip_ref:
        zip_ref.extractall(FOLDER_NAME)

    if not os.path.exists(FOLDER_NAME):
        print("❌ Failed to extract the archive.")
        print(f"Please download the archive manually from {DOWNLOAD_URL} and extract it yourself.")
        print("If the archive has been already downloaded, please unzip it manually.")
        exit(1)

    # ==== MOVE FOLDERS TO ROOT ====
    success = False
    for folder in ["weights", "data"]:
        src = os.path.join(FOLDER_NAME, folder)
        dst = os.path.join(".", folder)
        if os.path.exists(dst):
            print(f"⚠️ Folder '{folder}' already exists in the root. Skipping move.")
        elif os.path.exists(src):
            shutil.move(src, dst)
            print(f"✅ Moved '{folder}/' to root.")
            success = True
        else:
            print(f"❌ Folder '{folder}/' not found in archive.")
            print(f"Something went wrong, please download the archive manually from {DOWNLOAD_URL} and extract it yourself.")

    if success:
        # ==== CLEANUP ====
        print("🧹 Cleaning up...")
        os.remove(ZIP_NAME)
        shutil.rmtree(FOLDER_NAME)

        print("🎉 Done! Required folders are now in the root directory.")
