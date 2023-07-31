import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import socket
import requests
from tldextract import tldextract
import os
import numpy as np
import whois # pip install python-whois
from datetime import datetime
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import random
os.environ['ipinfo_token'] = open('./datasets/ipinfo_token.txt').read()
os.environ['webshrinker_token'] = open('./datasets/webshrinker_token.txt').read()
from PIL import Image, ImageDraw, ImageFont

def draw_annotated_image_nobox(image: Image.Image, txt: str):
    # Convert the image to RGBA for transparent overlay
    image = image.convert('RGBA')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load a larger font for text annotations
    font = ImageFont.truetype(font="./selection_model/fonts/arialbd.ttf", size=18)

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
    light_red_t = (128, 0, 0, 255)  # RGBA

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

class BrandAnalysis():
    def __init__(self, url_list, brand_list):
        self.url_list = url_list
        self.brand_list = brand_list

    @staticmethod
    def classify_url(url):
        BASE_URL = 'https://api.webshrinker.com/categories/v3/'

        """Classify a URL using the Webshrinker API."""
        headers = {
            'Authorization': f'Bearer {os.getenv("webshrinker_token")}'
        }
        response = requests.get(f'{BASE_URL}{url}', headers=headers)
        data = response.json()

        # Extract the primary category (You can modify this as per your needs)
        category = data.get('data', [{}])[0].get('categories', ['Unknown'])[0]
        return category

    def visualize_brands(self):
        brand_counts = Counter(self.brand_list)
        # Sort brands by frequency in descending order
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        brands_sorted = [item[0] for item in sorted_brands]
        counts_sorted = [item[1] for item in sorted_brands]

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

    def visualize_sectors(self):
        '''Visualize brand sectors'''
        """Generate a pie chart of sectors from a list of URLs."""
        # Classify each URL
        sectors = [self.classify_url(url) for url in self.url_list]

        # Aggregate the sectors
        sector_counts = {}
        for sector in sectors:
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

        # Visualize the results
        labels = list(sector_counts.keys())
        sizes = list(sector_counts.values())

        plt.figure(figsize=(10, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
        plt.title('Distribution of URLs by Sector')
        plt.savefig('./debug.png')

class IPAnalysis():

    def __init__(self, url_list):
        self.url_list = url_list

    @staticmethod
    def resolve_urls_to_ips(urls):
        ip_dict = {}
        for url in urls:
            try:
                ip_addresses = socket.getaddrinfo(url, None, proto=socket.IPPROTO_TCP)
                unique_ips = set([addr[4][0] for addr in ip_addresses])
                ip_dict[url] = list(unique_ips)
            except socket.gaierror:
                ip_dict[url] = []
        return ip_dict

    @staticmethod
    def get_ip_geolocation(ip):
        access_token = os.getenv("ipinfo_token")
        url = f"https://ipinfo.io/{ip}/json?token={access_token}"
        # Make the request
        response = requests.get(url)
        data = response.json()
        print(data)
        return data['country']

    def geoplot(self):
        ip_dict = self.resolve_urls_to_ips(self.url_list)
        ip_list = []
        for k, v in ip_dict.items():
            ip_list.extend(v)
        country_list = []
        for ip in ip_list:
            response = self.get_ip_geolocation(ip)
            country_list.append(response)

        country_counts = Counter(country_list)

        # Load the world map
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # Filter out Antarctica from the world dataset
        world = world[world['continent'] != "Antarctica"]

        # Add a new column to the dataframe for IP counts, default to 0
        world['ip_counts'] = world['name'].map(country_counts).fillna(0)

        # Create a custom black and white colormap
        cmap_bw = mcolors.LinearSegmentedColormap.from_list(
            "custom bw", [(1, 1, 1), (0, 0, 0)], N=256
        )
        # Set up a new figure with modified aspect ratio
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        # Remove axis and set title with better typography
        ax.axis('off')
        # Plot world boundaries in black
        world.boundary.plot(ax=ax, color='black', linewidth=0.5)

        # Plot heatmap using the black and white colormap
        world_map = world.plot(column='ip_counts', cmap=cmap_bw, linewidth=0.8, ax=ax, edgecolor='0.8',
                               legend=True, legend_kwds={'orientation': "horizontal", 'label': "IP counts", 'shrink': 0.9, 'pad': 0.01})

        # Adjust the aspect ratio of the plot
        ax.set_aspect('equal', adjustable='datalim')

        # Fine-tuned layout
        plt.tight_layout(pad=0.1)
        plt.savefig('./debug.png')

class CampaignAnalysis():
    def __init__(self, screenshot_path_list):
        self.screenshot_path_list = screenshot_path_list

    #TODO continuous request the status of the webpage
    @staticmethod
    def get_status_code(url):
        try:
            response = requests.get(url)
            return response.status_code
        except requests.RequestException as e:
            print(f"Error fetching the URL {url}: {e}")
            return None

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
    pass
