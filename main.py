import ee
import geemap
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import rasterio

#CONFIG
'''
Train=true means it fetches data from GEE and saves GeoTiffs to disk
Train=false means it loads saved files from disk and produces visualisations
'''
TRAIN = False 

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#INITIALIZE GEE
ee.Initialize(project='inbound-lattice-403009')
#STUDY AREA: Central Kalimantan, Borneo, Indonesia
# Define the study area (example: a rectangle around a specific location)
ROI = ee.Geometry.Rectangle([116.0, 0.5, 118.5, 2.5]) #order goes [west, south, east, north]
print(f"ROI: {ROI.getInfo()['coordinates']}")

YEAR_BEFORE = 2017
YEAR_AFTER = 2023

#FUNCTIONS
def get_imagery(year, roi):
    '''
    Loading cloud-masked imagery median composite for a given year and region of interest.
    '''
    start = f"{year}-01-01" 
    end = f"{year}-12-31"

    def mask_landsat_clouds(image):
        qa = image.select("QA_PIXEL") #QA_PIXEL is a quality flag band in Landsat imagery.
        cloud_mask = qa.bitwiseAnd(1 << 3).eq(0) #bit 3 is cloudflag in Landsat 8, so we create a mask where this bit is not set (i.e., cloud-free pixels).
        '''
        Landsat 8 stores pixel values same as integers, just like Sentinal - 2
        But the scaling formula is different — we multiply by 0.0000275 and subtract 0.2. 
        This is the official USGS formula for scaling to convert raw image to surface reflectance values between 0 and 1. 
        For example, a raw pixel value of 5000 would be scaled to 0.1375, meaning 13.75% reflectance. 
        This scaling is done after masking out clouds to avoid unnecessary computation on cloudy pixels.
        '''
        return image.updateMask(cloud_mask).select(["SR_B2","SR_B3","SR_B4","SR_B5","SR_B6","SR_B7"]).multiply(0.0000275).add(-0.2) #scale and offset for reflectance values
    
    def mask_s2_clouds(image):
        # QA60 is like a checklist attached to each pixel saying "is this pixel trustworthy or not."
        qa = image.select("QA60") #QA60 is a quality flag band in Sentinel-2 imagery.
        cloud_bit_mask = 1 << 10 #bit 10 is clouds
        cirrus_bit_mask = 1 << 11 #bit 11 is cirrus clouds
        mask = (
            qa.bitwiseAnd(cloud_bit_mask).eq(0)
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        )
        '''
        As raw Sentinal 2 images pixel values are stored as integers (e.g., 0-10000), 
        we need to divide by 10000 to get reflectance values in the range of 0-1. For example, 
        a pixel value of 5000 in the raw image would be 0.5, meaning 50% reflectance. This is done after computation so we do not 
        waste computation on pixels we already removed as clouds.
        '''
        return image.updateMask(mask).divide(10000) 
    if year <= 2018:
        #Use landsat 8 for years before 2019 as Sentinel-2 data is not available before that
        collection = (
            ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", 80)) #filter out images with more than 30% cloud cover
            .map(mask_landsat_clouds)
        )
        count = collection.size().getInfo()
        print(f"  Found {count} images for {year}")
        if count == 0:
            raise ValueError(f"No images found for year {year}. Try a different ROI or year.")
        return collection.median().clip(roi)
    
    #median = collection.median().clip(roi) #take the median value of each pixel across all images in the collection to create a single composite image for the year. 
                                         #This helps to reduce noise and fill in gaps caused by clouds.
    else:
        # Use Sentinel-2 for recent years
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
            .map(mask_s2_clouds)
        )
        return collection.median().clip(roi)
    # Fill missing bands with zeros to avoid empty image errors
    # band_names = ["B2", "B3", "B4", "B8"]
    # filled = median.unmask(0).select(band_names)
    # return filled

def compute_ndvi(image):
    band_names = image.bandNames().getInfo()
    '''
    Compute the Normalized Difference Vegetation Index (NDVI) for Sentinal-2 image.
    Bands 4(665nm) and 8(842nm) correspond to the red and near-infrared (NIR) bands, respectively.
    Healthy vegetation reflects more NIR and less red light, resulting in higher NDVI values.
    Bare soil, clouds, urban areas reflect red light strongly which means high band 4 values.

    So healthy vegetation has high Band 8 values.
    Water, bare soil, urban areas absorb NIR means low Band 8 values
    Bands 4 and 8 are the standard for NDVI because:

    They capture the exact spectral region where vegetation contrast is maximum
    Every vegetation index paper uses these two — it's the established standard

    '''
    if "B8" in band_names:
        #Sentinal-2 has B8 as NIR and B4 as red
        return image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    else:
        #Landsat 8 has B5 as NIR and B4 as red
        return image.normalizedDifference(["SR_B5", "SR_B4"]).rename("NDVI")

def classify_land_cover(ndvi_image):
    '''
    Simple threshold-based land cover classification based on NDVI values.
    Returns classified images with class labels: 
    water/bare soil (NDVI<=0.1) = 0, 
    degraded/sparse veg (NDVI>0.1 & NDVI<=0.3) = 1, 
    agriculture/young plantations (NDVI>0.3 & NDVI<=0.5) = 2, 
    forest (NDVI>0.5) = 3.

    '''
    water = ndvi_image.lt(0.1).multiply(0)
    degraded = ndvi_image.gte(0.1).And(ndvi_image.lte(0.3)).multiply(1)
    agri = ndvi_image.gte(0.3).And(ndvi_image.lt(0.5)).multiply(2)
    forest = ndvi_image.gte(0.5).multiply(3)

    return water.add(degraded).add(agri).add(forest).rename("class")

# main pipeline
if TRAIN:
    print("Fetching Sentinel-2 images ")
    img_before = get_imagery(YEAR_BEFORE, ROI)
    img_after = get_imagery(YEAR_AFTER, ROI)

    print("Computing NDVI...")
    ndvi_before = compute_ndvi(img_before)
    ndvi_after = compute_ndvi(img_after)

    print("Classifying land cover...")
    class_before = classify_land_cover(ndvi_before) 
    class_after = classify_land_cover(ndvi_after)

    #Change Detection: Subtract the classified images to identify changes in land cover between the two years.
    change = class_before.subtract(class_after).rename("change")

    #geemap.ee_export_image - Downloads an image from Google Earth Engine and saves it as a GeoTIFF file in the local machine.
    #here spatial res is set to 500x500m area per pixel to reduce file size and processing time. 
    #The region parameter ensures that only the area of interest is exported, 
    #and file_per_band=False means that all bands will be saved in a single multi-band GeoTIFF file rather than separate files for each band.
    geemap.ee_export_image(
        ndvi_before, filename=f"{OUTPUT_DIR}/ndvi_{YEAR_BEFORE}.tif",
        scale=500, region=ROI, file_per_band=False
    )

    geemap.ee_export_image(
        ndvi_after, filename=f"{OUTPUT_DIR}/ndvi_{YEAR_AFTER}.tif",
        scale=500,region=ROI, file_per_band=False
    )

    geemap.ee_export_image(
        class_before, filename=f"{OUTPUT_DIR}/class_{YEAR_BEFORE}.tif",
        scale=500,region=ROI, file_per_band=False
    )

    geemap.ee_export_image(
        class_after, filename=f"{OUTPUT_DIR}/class_{YEAR_AFTER}.tif",
        scale=500,region=ROI, file_per_band=False
    )

    geemap.ee_export_image(
        change, filename=f"{OUTPUT_DIR}/change_{YEAR_BEFORE}_{YEAR_AFTER}.tif",
        scale=500, region=ROI, file_per_band=False
    )
    print("Export Complete. Set TRAIN=FALSE to visualise results.")

else:
    #Load the saved GeoTIFF files from disk and produce visualisations
    print("Loading saved outputs for visualaisation...")
    files = {
        "ndvi_before": f"{OUTPUT_DIR}/ndvi_{YEAR_BEFORE}.tif", 
        "ndvi_after": f"{OUTPUT_DIR}/ndvi_{YEAR_AFTER}.tif",
        "class_before": f"{OUTPUT_DIR}/class_{YEAR_BEFORE}.tif", 
        "class_after": f"{OUTPUT_DIR}/class_{YEAR_AFTER}.tif",
        "change": f"{OUTPUT_DIR}/change_{YEAR_BEFORE}_{YEAR_AFTER}.tif"
    }

    # Check if any files are missing
    missing = [k for k, v in files.items() if not os.path.exists(v)]
    if missing:
        print(f"Missing files: {missing}")
        print("Run with TRAIN = True first to generate outputs.")
        exit()

    def read_tif(path):
        with rasterio.open(path) as src: 
            return src.read(1) #read the first band of the GeoTIFF file
        
    ndvi_before  = read_tif(files["ndvi_before"])
    ndvi_after   = read_tif(files["ndvi_after"])
    class_before = read_tif(files["class_before"])
    class_after  = read_tif(files["class_after"])
    change       = read_tif(files["change"])


    #  PLOT 
    class_cmap = mcolors.ListedColormap(["#4a90d9", "#d4a017", "#a8d08d", "#2d6a4f"])
    class_norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], class_cmap.N)
    class_labels = ["Water/Bare Soil", "Degraded", "Agriculture", "Forest"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Borneo Land Use Change Detection (Kalimantan)\n"
                f"{YEAR_BEFORE} - {YEAR_AFTER} | Landsat 8 ({YEAR_BEFORE}) & Sentinel-2 ({YEAR_AFTER})",
                fontsize=14, fontweight="bold")
    # NDVI before
    im0 = axes[0, 0].imshow(ndvi_before, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0, 0].set_title(f"NDVI {YEAR_BEFORE}")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # NDVI after
    im1 = axes[0, 1].imshow(ndvi_after, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0, 1].set_title(f"NDVI {YEAR_AFTER}")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # NDVI difference
    ndvi_diff = ndvi_after - ndvi_before
    im2 = axes[0, 2].imshow(ndvi_diff, cmap="RdYlGn", vmin=-0.4, vmax=0.4)
    axes[0, 2].set_title("NDVI Change")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)

    # Class before
    im3 = axes[1, 0].imshow(class_before, cmap=class_cmap, norm=class_norm)
    axes[1, 0].set_title(f"Land Cover {YEAR_BEFORE}")
    cbar3 = plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar3.set_ticklabels(class_labels)

    # Class after
    im4 = axes[1, 1].imshow(class_after, cmap=class_cmap, norm=class_norm)
    axes[1, 1].set_title(f"Land Cover {YEAR_AFTER}")
    cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar4.set_ticklabels(class_labels)

    # Change map
    change_cmap = mcolors.ListedColormap(["#d73027", "#fee08b", "#1a9850"])
    change_norm = mcolors.BoundaryNorm([-3.5, -0.5, 0.5, 3.5], change_cmap.N)
    im5 = axes[1, 2].imshow(change, cmap=change_cmap, norm=change_norm)
    axes[1, 2].set_title("Change Map")
    cbar5 = plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, ticks=[-1, 0, 1])
    cbar5.set_ticklabels(["Degraded", "No Change", "Improved"])

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/land_use_change_map.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUTPUT_DIR}/land_use_change_map.png")

    #STATS 
    print("\n Land Cover Statistics ")
    total = class_before.size
    for i, label in enumerate(class_labels):
        before_pct = np.sum(class_before == i) / total * 100
        after_pct  = np.sum(class_after  == i) / total * 100
        print(f"{label:20s}  {YEAR_BEFORE}: {before_pct:5.1f}%  -  {YEAR_AFTER}: {after_pct:5.1f}%")

'''
The classification values are there but the displayed values are wrong. the classified GeoTIFF values are
stored as floats but the colormap boundary norm is set to integers 0-3. 
 if a pixel's class number went down between 2017 and 2023 (e.g. forest → degraded), 
 the subtraction gives a positive number → red on the map. If it went up (recovery), negative number → green.

'''