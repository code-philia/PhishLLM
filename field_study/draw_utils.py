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
import os
from tqdm import tqdm
from skimage import io, transform
from skimage.metrics import structural_similarity as ssim
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
from operator import itemgetter
from datetime import datetime, timedelta
import seaborn as sns

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
        plt.title('Frequency of Phishing Targets')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(False)  # Turn off the grid
        plt.savefig('./field_study/brand_freq.png')

    def visualize_sectors(sectors, threshold=2.0):
        '''Visualize brand sectors'''

        # Aggregate the sectors
        sector_counts = {}
        for sector in sectors:
            if sector == 'None':
                sector = 'Uncategorized'
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Remove 'Uncategorized' class if present
        if 'Uncategorized' in sector_counts:
            del sector_counts['Uncategorized']

        # Combine small percentages into "Other" category
        total = sum(sector_counts.values())
        other_count = 0
        for sector, count in list(sector_counts.items()):
            if 100.0 * count / total < threshold:
                other_count += count
                del sector_counts[sector]
        if other_count > 0:
            sector_counts['Other'] = other_count

        # Visualize the results
        labels = list(sector_counts.keys())
        sizes = list(sector_counts.values())

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = sns.color_palette("husl", len(labels))  # Professional color palette
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=None,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            pctdistance=0.85,
            wedgeprops={'edgecolor': 'grey'},
        )

        # Draw a circle at the center to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        # Annotate each pie slice with corresponding label and draw a line
        for i, (w, t) in enumerate(zip(wedges, texts)):
            ang = (w.theta2 - w.theta1) / 2. + w.theta1
            y = np.sin(np.deg2rad(ang))
            x = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                        horizontalalignment=horizontalalignment,
                        fontsize=10, weight="bold",
                        arrowprops=dict(arrowstyle="-", connectionstyle=connectionstyle))

        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
        plt.title('Distribution of Phishing Targets by Sector')
        plt.tight_layout()
        plt.savefig('./field_study/brand_sector.png')

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
        plt.title('Geolocation Distribution of Phishing IPs')
        plt.savefig('./field_study/geo.png')

class CampaignAnalysis:
    @staticmethod
    def similarity_threshold_clustering(similarity_matrix, threshold):
        # Create a graph from the similarity matrix
        G = nx.Graph()
        n = len(similarity_matrix)

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j] >= threshold:
                    G.add_edge(i, j)

        # Use NetworkX's connected_components function to get the clusters
        clusters = list(nx.connected_components(G))

        return clusters

    @staticmethod
    def cluster_to_timeseries(cluster, all_dates):
        # Extract the dates from the cluster and count the number of screenshots for each date.
        dates = [screenshot_date for screenshot, screenshot_date in cluster]
        date_counts = Counter(dates)

        # Fill in any missing dates with zero counts.
        full_counts = [date_counts.get(date, 0) for date in all_dates]

        # Compute the cumulative counts.
        cumulative_counts = np.cumsum(full_counts)

        return all_dates, cumulative_counts

    def cluster_shot_representations(self, shot_path_list):
        # Load and resize all images.
        model = models.resnet50(pretrained=True)
        model = model.eval()

        # Use the layer before the final fully-connected layer for feature extraction.
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

        # Define the image transforms.
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Extract features for all images.
        features = []
        for image_path in shot_path_list:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)
            with torch.no_grad():
                feature = feature_extractor(image).squeeze().numpy()
            features.append(feature)

        # Compute pairwise distances between the features.
        similarity_matrix = cosine_similarity(features)
        clusters = self.similarity_threshold_clustering(similarity_matrix, 0.99)

        clusters_path = []
        for it, cls in enumerate(clusters):
            shots_under_cls = [(shot_path_list[ind], os.path.basename(os.path.dirname(os.path.dirname(shot_path_list[ind])))) for ind in cls]
            clusters_path.append(shots_under_cls)
        return clusters_path

    def visualize_campaign(self, clusters):

        # Create a new figure.
        plt.figure(figsize=(10, 6))

        # Use a professional color palette
        colors = sns.color_palette("husl", len(clusters))

        # Collect all unique dates
        all_dates = set()
        for cluster in clusters:
            _, dates = zip(*cluster)
            all_dates.update(dates)
        all_dates = sorted(list(all_dates), key=lambda date: datetime.strptime(date, '%Y-%m-%d'))

        # Convert each cluster into a timeseries and plot it.
        for i, (cluster, color) in enumerate(zip(clusters, colors)):
            # Skip clusters with fewer than 3 items or only seen on a single date
            _, dates = zip(*cluster)
            if len(cluster) < 4 or len(set(dates)) == 1:
                continue

            dates, counts = self.cluster_to_timeseries(cluster, all_dates)
            plt.plot(dates, counts, marker='o', color=color, label=f'Cluster {i + 1}')

        # Add a legend and labels.
        plt.xticks(range(len(all_dates)), all_dates)
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Cumulative number of screenshots')
        plt.title('Phishing Campaign over Time')

        plt.tight_layout()
        plt.grid(True)  # Adding a grid for better readability
        sns.despine(left=True, bottom=True)  # Remove the top and right spines

        # Save the figure.
        plt.savefig('./field_study/campaign.png')

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
    clusters_path = campaign.cluster_shot_representations(shot_path_list)
    print(clusters_path)
    campaign.visualize_campaign(clusters_path)