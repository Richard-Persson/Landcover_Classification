import rasterio
import matplotlib.pyplot as plt
import numpy as np

IMAGE = "data/raw/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"


with rasterio.open(IMAGE) as src:
     img = src.read()  # Leser alle bånd som en NumPy-array
     print(img.shape)  # (bands, height, width)


# Assuming NIR is band 7 and red is band 3
nir_band = img[7]  # NIR band (for example)
red_band = img[2]  # Red band

# Calculate NDVI
ndvi = (nir_band - red_band) / (nir_band + red_band)

# Plot the NDVI result
plt.imshow(ndvi, cmap="RdYlGn")
plt.title("NDVI Visualization")
plt.colorbar()
plt.show()

num_bands = 13
num_columns = 5  # You can adjust this if you want more or fewer columns
num_rows = (num_bands // num_columns) + (num_bands % num_columns > 0)  # Compute required rows

# Plot all 13 bands in a grid
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 10))  # Create the grid dynamically
axes = axes.ravel()  # Flatten the 2D array of axes into 1D

for i in range(13):
    ax = axes[i]
    ax.imshow(img[i], cmap="viridis")  # Apply a colormap for better contrast
    ax.set_title(f"Band {i+1}")
    ax.axis("off")  # Turn off axis labels for cleaner visualization

# Hide the remaining empty axes (if any)
for i in range(num_bands, len(axes)):
    axes[i].axis("off")

plt.tight_layout()
plt.show()
