import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import socket
import requests
from tldextract import tldextract
import os
import numpy as np
import whois # pip install python-whois
from datetime import datetime, date
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import random
from PIL import Image, ImageDraw, ImageFont
from field_study.monitor_url_status import *
from field_study.results_statistics import get_pos_site, daterange
import numpy as np
from skimage.transform import rescale, resize
import os
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fcluster

def draw_annotated_image_nobox(image: Image.Image, txt: str):
    # Convert the image to RGBA for transparent overlay
    image = image.convert('RGBA')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load a larger font for text annotations
    font = ImageFont.truetype(font="./selection_model/fonts/arialbd.ttf", size=25)

    # Calculate the width and height of the text
    text_width, text_height = draw.textsize("Output: "+txt, font=font)

    # Create an image with extra space at the bottom for the text
    new_height = image.height + text_height + 10  # 10 for padding
    final_image = Image.new('RGBA', (image.width, new_height), (255, 255, 255, 255))
    final_image.paste(image, (0, 0))

    draw_final = ImageDraw.Draw(final_image)
    draw_final.text((10, image.height + 5), "Output: "+txt, font=font, fill="black")
    return final_image

def draw_annotated_image(image: Image.Image, boxes: list, txts: list, scores: list, crop_size=(1000, 600)):

    # Ensure that boxes, txts, and scores have the same length
    assert len(boxes) == len(txts) == len(scores), "boxes, txts, and scores must have the same length"

    # Convert the image to RGBA for transparent overlay
    image = image.convert('RGBA')

    left_margin = (image.width - crop_size[0]) / 2
    top_margin = (image.height - crop_size[1]) / 2
    right_margin = left_margin + crop_size[0]
    bottom_margin = top_margin + crop_size[1]
    for i, box in enumerate(boxes):
        boxes[i] = [[max(coord[0] - left_margin, 0), max(coord[1] - top_margin, 0)] for coord in box]

    image = image.crop((left_margin, top_margin, right_margin, bottom_margin))

    # Create a temporary RGBA image to draw on
    tmp = Image.new('RGBA', image.size, (0, 0, 0, 0))

    # Create a drawing context
    draw = ImageDraw.Draw(tmp)

    # Load a larger font for text annotations
    font = ImageFont.truetype(font="./selection_model/fonts/arialbd.ttf", size=30)

    # Define light red color with 80% transparency
    light_red = (128, 0, 0, int(0.4 * 255))  # RGBA

    for box, txt, score in zip(boxes, txts, scores):
        # Draw the bounding box with 80% transparent fill
        draw.polygon([
            tuple(box[0]),
            tuple(box[1]),
            tuple(box[2]),
            tuple(box[3])
        ], outline="red", fill=light_red, width=3)

        # Calculate text position to be at the right of the box
        text_width, text_height = draw.textsize(txt, font=font)
        text_x = box[1][0] + 15
        text_y = (box[1][1] + box[2][1]) / 2 - text_height / 2

        # Annotate the text
        draw.text((text_x, text_y), txt, font=font, fill="red", width=3)

    # Combine the original image and the temporary image
    result = Image.alpha_composite(image, tmp)

    # Concatenate all texts and add below the image
    combined_text = 'Output: \n' + ' '.join(txts)
    font = ImageFont.truetype(font="./selection_model/fonts/arialbd.ttf", size=18)
    text_width, text_height = draw.textsize(combined_text, font=font)

    # Create an image with extra space at the bottom for the concatenated text
    new_height = result.height + text_height + 10  # 10 for padding
    final_image = Image.new('RGBA', (result.width, new_height), (255, 255, 255, 255))
    final_image.paste(result, (0, 0))

    draw_final = ImageDraw.Draw(final_image)
    draw_final.text((10, result.height + 5), combined_text, font=font, fill="black")

    return final_image


class BrandAnalysis:

    def visualize_brands(brand_list):
        brand_counts = Counter(brand_list)
        del brand_counts['firezone.com']

        # Sort brands by frequency in descending order
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        brands_sorted = [item[0] for item in sorted_brands][:5]
        counts_sorted = [item[1] for item in sorted_brands][:5]

        color = 'lightgray'
        plt.figure(figsize=(12, 6))
        plt.bar(brands_sorted, counts_sorted, color=color, edgecolor='black')
        plt.xlabel('Brands')
        plt.ylabel('Number of Times Targeted')
        plt.title('Frequency of Brands being Targeted for Phishing')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(False)  # Turn off the grid
        plt.savefig('./debug.png')

    def visualize_sectors(sectors):
        '''Visualize brand sectors'''

        # Aggregate the sectors
        sector_counts = {}
        for sector in sectors:
            if sector == 'None':
                sector = 'Uncategorized'
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Visualize the results
        labels = list(sector_counts.keys())
        sizes = list(sector_counts.values())

        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
        plt.title('Distribution of URLs by Sector')
        plt.savefig('./debug.png')

class IPAnalysis:

    def geoplot(coordinates):
        import geopandas as gpd
        from shapely.geometry import Point
        import cartopy.crs as ccrs  # Import the required module for the Robinson projection
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

        # Filter out Antarctica
        world = world[world['continent'] != "Antarctica"]

        # Count the occurrences of coordinates for each country
        country_points = []
        for coord in coordinates:
            country = world[world.geometry.contains(Point(coord[1], coord[0]))]['name'].values
            if len(country) > 0:
                country_points.append(country[0])
            else:
                country_points.append("Unknown")

        country_counts = Counter(country_points)
        if "Unknown" in country_counts:
            del country_counts["Unknown"]

        world['counts'] = world['name'].map(country_counts).fillna(0)

        fig, ax = plt.subplots(1, 1, figsize=(15, 10),
                               subplot_kw={'projection': ccrs.Robinson()})  # Set the projection to Robinson

        # Color countries based on counts
        world = world.to_crs(ccrs.Robinson().proj4_init)
        world.plot(column='counts', cmap='OrRd', edgecolor='#bbbbbb', legend=False, ax=ax)

        # Plot the provided coordinates
        x_coords = [coord[1] for coord in coordinates]
        y_coords = [coord[0] for coord in coordinates]
        ax.scatter(x_coords, y_coords, color='red', s=15, edgecolor='white', linewidth=0.5,
                   zorder=5, transform=ccrs.PlateCarree())  # Use PlateCarree for the scatter points

        # Enhancements
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#d9d9d9')  # Subtle grid

        plt.tight_layout()
        plt.savefig('./debug.png')

class CampaignAnalysis:

    def cache_shot_representations(self, shot_path_list):
        pass

    def cluster_shot_representations(self, shot_path_list):
        pass

    def visualize_campaign(self):
        start_date_simulation = datetime(2023, 1, 1)
        end_date_simulation = datetime(2023, 1, 31)

        phishing_data_simulation = [
            {
                'cluster_id': random.randint(1, 50),  # Assuming there are 50 potential clusters
                'first_seen': (start_date_simulation + timedelta(days=random.randint(0, 29))).strftime('%Y-%m-%d'),
                'last_seen': (start_date_simulation + timedelta(days=random.randint(1, 30))).strftime('%Y-%m-%d')
            }
            for _ in range(1000)
        ]

        # Convert simulated data to a DataFrame
        df_simulation = pd.DataFrame(phishing_data_simulation)
        df_simulation['first_seen'] = pd.to_datetime(df_simulation['first_seen'])
        df_simulation['last_seen'] = pd.to_datetime(df_simulation['last_seen'])

        # Filter out clusters with only one sample
        clusters_counts_simulation = df_simulation['cluster_id'].value_counts()
        valid_clusters = clusters_counts_simulation[clusters_counts_simulation > 1].index.tolist()

        df_simulation = df_simulation[df_simulation['cluster_id'].isin(valid_clusters)]

        # Create a timeline of dates
        timeline_simulation = pd.date_range(start_date_simulation, end_date_simulation)

        # Create a dictionary to store the count of phishing sites in each cluster for each date
        clusters = df_simulation['cluster_id'].unique()
        cluster_counts = {cluster: [0] * len(timeline_simulation) for cluster in clusters}
        total_counts = {cluster: sum(counts) for cluster, counts in cluster_counts.items()}
        top_5_clusters = sorted(total_counts, key=total_counts.get, reverse=True)[:5]

        # Plotting the data for only the top 5 clusters
        plt.figure(figsize=(12, 6))
        for cluster in top_5_clusters:
            plt.plot(timeline_simulation, cluster_counts[cluster], label=f'Cluster {cluster}', linewidth=1.5)
        plt.xlabel('Date')
        plt.ylabel('Number of Phishing Sites')
        plt.title('Top 5 Phishing Campaigns Over Time')
        plt.legend(frameon=True, loc='upper left')
        plt.tight_layout()

        # Removing the right and top spines for a cleaner look
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.savefig('./debug.png')

if __name__ == '__main__':
    base = "/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples"
    '''geolocation'''
    gs_sheet = gwrapper_monitor()
    rows = gs_sheet.get_records()
    # geo_loc_list = list(map(lambda x: tuple(map(float, x['geo_loc'].split(','))), filter(lambda x: x['geo_loc'] != 0, rows)))
    # IPAnalysis.geoplot(geo_loc_list)

    '''brand'''
    # brands = list(map(lambda x: x['brand'], rows))
    # BrandAnalysis.visualize_brands(brands)

    '''sector'''
    # sectors = list(map(lambda x: x['sector'], rows))
    # BrandAnalysis.visualize_sectors(sectors)

    '''phishing campaign'''
    shot_path_list = list(map(lambda x: os.path.join(base, x['date'], x['foldername'], 'shot.png'), rows))
    campaign = CampaignAnalysis()
    campaign.cluster_shot_representations(shot_path_list)