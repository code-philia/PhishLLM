from datetime import datetime, date
from PIL import Image, ImageDraw, ImageFont
from field_study.monitor_url import *
from field_study.results_statistics import get_pos_site, daterange
import os
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import seaborn as sns
from typing import List
import cv2

def draw_annotated_image_nobox(image: Image.Image, txt: str):
    # Convert the image to RGBA for transparent overlay
    image = image.convert('RGBA')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load a larger font for text annotations
    font = ImageFont.truetype(font="./fonts/arialbd.ttf", size=25)

    # Calculate the width and height of the text
    text_width, text_height = draw.textsize("Output: "+txt, font=font)

    # Create an image with extra space at the bottom for the text
    new_height = image.height + text_height + 10  # 10 for padding
    final_image = Image.new('RGBA', (image.width, new_height), (255, 255, 255, 255))
    final_image.paste(image, (0, 0))

    draw_final = ImageDraw.Draw(final_image)
    draw_final.text((10, image.height + 5), "Output: "+txt, font=font, fill="black")
    return final_image

def draw_annotated_image_box(image: Image.Image, predicted_domain: str, box: List[float]):
    image = image.convert('RGB')
    screenshot_img_arr = np.asarray(image)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)
    screenshot_img_arr = screenshot_img_arr.astype(np.uint8)

    if box is not None:
        cv2.rectangle(screenshot_img_arr, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (69, 139, 0), 2)
        cv2.putText(screenshot_img_arr, 'Predicted phishing target: '+ predicted_domain, (int(box[0]), int(box[3])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 2)
    else:
        cv2.putText(screenshot_img_arr, 'Predicted phishing target: ' + predicted_domain, (int(10), int(10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 2)
    screenshot_img_arr = np.flip(screenshot_img_arr, -1)
    image = Image.fromarray(screenshot_img_arr)
    return image

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
    font = ImageFont.truetype(font="./fonts/arialbd.ttf", size=30)

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
    font = ImageFont.truetype(font="./fonts/arialbd.ttf", size=18)
    text_width, text_height = draw.textsize(combined_text, font=font)

    # Create an image with extra space at the bottom for the concatenated text
    new_height = result.height + text_height + 10  # 10 for padding
    final_image = Image.new('RGBA', (result.width, new_height), (255, 255, 255, 255))
    final_image.paste(result, (0, 0))

    draw_final = ImageDraw.Draw(final_image)
    draw_final.text((10, result.height + 5), combined_text, font=font, fill="black")

    return final_image

class GeneralAnalysis:
    def visualize_count(dates, phishLLM_counts, phishpedia_counts, phishIntention_counts):
        # Create a DataFrame
        data = {
            'Date': dates,
            'PhishLLM': phishLLM_counts,
            'Phishpedia': phishpedia_counts,
            'PhishIntention': phishIntention_counts
        }
        df = pd.DataFrame(data)

        # Set Seaborn style for an academic look
        sns.set_style("whitegrid")
        colors = sns.color_palette("pastel")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Width of a bar
        width = 0.25

        # Position of bars on x-axis
        r1 = np.arange(len(df['Date']))
        r2 = [x + width for x in r1]
        r3 = [x + width for x in r2]

        plt.bar(r1, df['PhishLLM'], width=width, color=colors[0], label='PhishLLM')
        plt.bar(r2, df['Phishpedia'], width=width, color=colors[1], label='Phishpedia')
        plt.bar(r3, df['PhishIntention'], width=width, color=colors[2], label='PhishIntention')

        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Phishing Reported', fontsize=12)
        plt.title('Daily Phishing Reports by Solution', fontsize=14)
        plt.xticks([r + width for r in range(len(df['Date']))], df['Date'], rotation=45)
        plt.yticks(np.arange(0, max(df['PhishLLM'].max(), df['Phishpedia'].max(), df['PhishIntention'].max()) + 1, 1))
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

        plt.tight_layout()
        plt.savefig('./field_study/plots/num_phish.png')
        plt.close()

class DomainAnalysis:
    def tld_distribution(domain_list):
        tld_list = list(map(lambda x: tldextract.extract(x).suffix, domain_list))
        tld_counts = Counter(tld_list)

        # Get the top 5 most frequent TLDs
        top_5_tlds = tld_counts.most_common(5)

        # Print the result or return it as needed
        print("Top 5 frequently used top-level domains:")
        for tld, count in top_5_tlds:
            print(f"{tld}: {count} occurrences")

    def domain_age_distribution(domain_age_list):
        sns.set_style("whitegrid")
        plt.hist(domain_age_list, bins=20, edgecolor='black', color='lightblue', alpha=0.7)
        plt.xlabel('Domain Age (in years)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Domain Ages')
        plt.xlim(left=0)
        plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        plt.savefig('./field_study/plots/domain_age.png')
        plt.close()

class BrandAnalysis:

    def visualize_brands(brand_list):
        brand_counts = Counter(brand_list)

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
        plt.yticks(np.arange(0, max(counts_sorted) + 1, 1))  # Set ytick labels at integer values
        plt.tight_layout()
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(False)  # Turn off the grid
        plt.savefig('./field_study/plots/brand_freq.png')
        plt.close()

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
        colors = sns.color_palette("husl", len(labels))
        if 'Other' in labels:
            other_index = labels.index('Other')
            colors[other_index] = 'gray'

        wedges, texts, autotexts = ax.pie(
            sizes,
            autopct='%1.1f%%',
            startangle=140,
            colors=colors,
            pctdistance=0.85,  # Adjust this value to position the percentage labels
            wedgeprops={'edgecolor': 'grey'},
        )

        # Draw a circle at the center to make it a donut chart
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)

        # Enhance the appearance of the percentage labels
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_weight('bold')

        # Add a legend with the sector labels
        plt.legend(
            loc="best",
            labels=labels,
            prop={'size': 10},
            title="Sectors",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )

        plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
        plt.title('Distribution of Phishing Targets by Sector')
        plt.tight_layout()
        plt.savefig('./field_study/plots/brand_sector.png')
        plt.close()

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
        plt.savefig('./field_study/plots/geo.png')
        plt.close()

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
            print(cluster)
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
        plt.savefig('./field_study/plots/campaign.png')
        plt.close()

if __name__ == '__main__':
    base = "/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples"
    '''geolocation'''
    gs_sheet = gwrapper_monitor()
    rows = gs_sheet.get_records()
    geo_loc_list = list(map(lambda x: tuple(map(float, x['geo_loc'].split(','))), filter(lambda x: x['geo_loc'] != 0, rows)))
    IPAnalysis.geoplot(geo_loc_list)

    '''phishing counts over time'''
    start_date = date(2023, 8, 7)
    today = datetime.today().date()
    # end_date = today + timedelta(days=1)
    end_date = today
    dates = []
    llm_counts = []
    pedia_counts = []
    intention_counts = []

    for single_date in daterange(start_date, end_date):
        date_ = single_date.strftime("%Y-%m-%d")
        llm_pos = get_pos_site('./field_study/results/{}_phishllm.txt'.format(date_))
        pedia_pos = get_pos_site('./field_study/results/{}_phishpedia.txt'.format(date_))
        intention_pos = get_pos_site('./field_study/results/{}_phishintention.txt'.format(date_))
        dates.append(date_)
        llm_counts.append(len(llm_pos))
        pedia_counts.append(len(pedia_pos))
        intention_counts.append(len(intention_pos))

    '''# of phishing over time'''
    GeneralAnalysis.visualize_count(dates, llm_counts, pedia_counts, intention_counts)

    '''tld distribution domain age distribution'''
    domains = list(map(lambda x: x['foldername'], rows))
    domain_ages = list(map(lambda x: x['domain_age'], rows))
    DomainAnalysis.tld_distribution(domains)
    DomainAnalysis.domain_age_distribution(domain_ages)

    '''brand'''
    brands = list(map(lambda x: x['brand'], rows))
    BrandAnalysis.visualize_brands(brands)

    '''sector'''
    sectors = list(map(lambda x: x['sector'], rows))
    BrandAnalysis.visualize_sectors(sectors)

    '''phishing campaign'''
    shot_path_list = list(map(lambda x: os.path.join(base, x['date'], x['foldername'], 'shot.png'), rows))
    campaign = CampaignAnalysis()
    clusters_path = campaign.cluster_shot_representations(shot_path_list)
    campaign.visualize_campaign(clusters_path)

    # alexa_urls = [x.strip().split(',')[1] for x in open('./datasets/top-1m.csv').readlines()]
    # DomainAnalysis.tld_distribution(alexa_urls)

    # [('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-15/cloudmail.esit.info/shot.png', '2023-08-15'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-15/mailtest.ghbank.com.cn/shot.png', '2023-08-15'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-08/gate.marinaccountants.com.au/shot.png', '2023-08-08'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-16/device-f55e9000-4769-4454-b2cb-625104881f16.remotewd.com/shot.png', '2023-08-16'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-08/device-f5a5f76d-75d0-4257-a002-6ce21167d81a.remotewd.com/shot.png', '2023-08-08'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-09/mail.design-industrial.eu/shot.png', '2023-08-09'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-09/exch01.alsdorf.contecgmbh.com/shot.png', '2023-08-09'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-14/srv-ex2-10358.gtkp.de/shot.png', '2023-08-14'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-09/mailto.ec-verpackungsservice.de/shot.png', '2023-08-09'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-09/exch-rostecnpf.esit.info/shot.png', '2023-08-09'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-09/device-8b248998-6847-4def-9eb1-9b59fb283b04.remotewd.com/shot.png', '2023-08-09'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-14/zahnmedizin-rathaus.my3cx.de/shot.png', '2023-08-14'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-09/remote.weissert.info/shot.png', '2023-08-09'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-13/device-0f18bd64-8456-493c-b545-ff128bd8fbdd.remotewd.com/shot.png', '2023-08-13'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-17/email.fueger-gmbh.de/shot.png', '2023-08-17'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-17/ex13.consultic.info/shot.png', '2023-08-17'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-17/mail.consultic.info/shot.png', '2023-08-17'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-10/webmail.trenker-3tconsulting.com/shot.png', '2023-08-10'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-11/headoffice1.travid.org/shot.png', '2023-08-11'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-15/secure.mawsonwest.com/shot.png', '2023-08-15')]
    # [('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-12/luka.sui.ducoccho1.click/shot.png', '2023-08-12'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-09/www.lf.wuangu1.click/shot.png', '2023-08-09'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-15/webfb.anhlongvedithoi.click/shot.png', '2023-08-15'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-08/login-usa.xuanbac.click/shot.png', '2023-08-08'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-13/zuk.pergugu.click/shot.png', '2023-08-13'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-08/login-france.xuanbac.click/shot.png', '2023-08-08')]
    # [('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-15/596.vscustomer.com/shot.png', '2023-08-15'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-11/1116.vscustomer.com/shot.png', '2023-08-11'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-15/235.vscustomer.com/shot.png', '2023-08-15'), ('/home/ruofan/git_space/ScamDet/datasets/phishing_TP_examples/2023-08-15/35.vscustomer.com/shot.png', '2023-08-15')]
