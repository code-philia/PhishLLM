from datetime import datetime, date
from PIL import Image, ImageDraw, ImageFont
from experiments.field_study.monitor_url import *
from experiments.field_study.results_statistics import get_pos_site, daterange
import os
import networkx as nx
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime, timedelta
import seaborn as sns
from typing import List
import cv2
from Levenshtein import distance as levenshtein_distance
from itertools import cycle
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def draw_annotated_image_nobox(image: Image.Image, txt: str):
    # Convert the image to RGBA for transparent overlay
    image = image.convert('RGBA')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load a larger font for text annotations
    font = ImageFont.truetype(font="./field_study/fonts/arialbd.ttf", size=25)

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
    font = ImageFont.truetype(font="./field_study/fonts/arialbd.ttf", size=30)

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
    font = ImageFont.truetype(font="./field_study/fonts/arialbd.ttf", size=18)
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

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))

        # Width of a bar
        width = 0.25

        # Position of bars on x-axis
        r1 = np.arange(len(df['Date']))
        r2 = [x + width for x in r1]
        r3 = [x + width for x in r2]

        # Create bars with different fill patterns
        plt.bar(r1, df['PhishLLM'], width=width, color='grey', edgecolor='black', label='PhishLLM')
        plt.bar(r2, df['Phishpedia'], width=width, color='lightgrey', edgecolor='black', label='Phishpedia')
        plt.bar(r3, df['PhishIntention'], width=width, color='darkgrey', edgecolor='black', label='PhishIntention')

        # Labels and title
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Number of Phishing Reported', fontsize=12)

        # X-axis ticks
        plt.xticks([r + width for r in range(len(df['Date']))], df['Date'], rotation=90)

        # Adding grid
        ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)

        # Adding the legend
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

        # Layout adjustment
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
        fig, ax = plt.subplots(figsize=(16, 6))  # Wider figure to match the second code

        sns.histplot(domain_age_list, bins=20, color='lightgray', edgecolor='black',
                     kde=False)  # Match color and edgecolor

        plt.xlabel('Domain Age (in years)', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.7)  # Light grid lines only for y-axis

        plt.xlim(left=0)

        plt.gca().spines['top'].set_visible(False)  # Remove top spine
        plt.gca().spines['right'].set_visible(False)  # Remove right spine

        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.savefig('./field_study/plots/domain_age.png')
        plt.close()

class BrandAnalysis:

    def visualize_brands(brand_list, topk=10):
        sns.set_style("whitegrid")
        brand_counts = Counter(brand_list)

        # Sort brands by frequency in descending order
        sorted_brands = sorted(brand_counts.items(), key=lambda x: x[1], reverse=True)
        brands_sorted = [item[0] for item in sorted_brands][:topk]
        counts_sorted = [item[1] for item in sorted_brands][:topk]

        plt.figure(figsize=(20, 10))  # Wider figure
        plt.bar(brands_sorted, counts_sorted, color='lightgray', edgecolor='black')
        plt.xlabel('Brands', fontsize=25)
        plt.ylabel('Number of Times Targeted', fontsize=25)
        plt.xticks(rotation=45, fontsize=25)
        plt.yticks(np.arange(0, max(counts_sorted) + 1, 1), fontsize=15)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')  # Light grid lines only for y-axis

        plt.tight_layout()
        plt.savefig('./field_study/plots/brand_freq.png')
        plt.close()

    def visualize_sectors(sectors, topk=5):
        # Aggregate the sectors
        sector_counts = Counter(sectors)

        # Calculate total count including 'None' and 'Uncategorized'
        total = sum(sector_counts.values())

        # Remove 'None' or 'Uncategorized' class for visualization
        sector_counts.pop('None', None)
        sector_counts.pop('Uncategorized', None)

        # Sort sectors by count
        sorted_sectors = sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:topk]

        # Calculate percentages
        sorted_percentages = [(label, (count / total) * 100) for label, count in sorted_sectors]

        # Visualize the results
        sns.set_style("whitegrid")
        labels, percentages = zip(*sorted_percentages)

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = sns.color_palette("Blues", len(labels))

        ax.barh(labels, percentages, color=colors, edgecolor='grey')
        ax.set_xlabel('Percentage (%)', fontsize=20)
        ax.set_ylabel('Sectors', fontsize=20)
        # ax.set_title('Top {} Phishing Targets by Sector'.format(topk), fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=15)

        sns.despine(left=True, bottom=True)
        plt.tight_layout()
        plt.savefig('./field_study/plots/brand_sector.png')
        plt.close()


class IPAnalysis:

    def geoplot(coordinates):
        import geopandas as gpd
        from shapely.geometry import Point
        import cartopy.crs as ccrs  # Import the required module for the Robinson projection
        from collections import Counter
        import matplotlib.pyplot as plt

        # Load the world map
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

        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(30, 10), subplot_kw={'projection': ccrs.Robinson()})

        # Color countries based on counts
        world = world.to_crs(ccrs.Robinson().proj4_init)
        world.plot(column='counts', cmap='OrRd', edgecolor='#bbbbbb', ax=ax)

        # Plot the provided coordinates
        x_coords = [coord[1] for coord in coordinates]
        y_coords = [coord[0] for coord in coordinates]
        ax.scatter(x_coords, y_coords, color='black', s=50, edgecolor='black', linewidth=0.5,
                   zorder=5, transform=ccrs.PlateCarree(), label='Phishing IP Locations')

        # Enhancements
        ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', linestyle='--')

        # Add legend for the scatter points
        scatter_legend = ax.legend(loc='lower left', fontsize=20, frameon=True, framealpha=1, edgecolor='black')
        scatter_legend.get_frame().set_facecolor('white')

        # Tight layout
        plt.tight_layout()

        # Save the plot
        plt.savefig('./field_study/plots/geo.png', dpi=300, bbox_inches='tight')
        plt.close()


class CampaignAnalysis:
    @staticmethod
    def similarity_threshold_clustering(mask, domain_name_list, threshold):
        G = nx.Graph()
        n = len(mask)

        for i in range(n):
            for j in range(i + 1, n):
                if mask[i, j] == 1:
                    domain1 = domain_name_list[i]
                    domain2 = domain_name_list[j]

                    subdomain1, subdomain2 = tldextract.extract(domain1).subdomain, tldextract.extract(domain2).subdomain
                    domain1, domain2 = tldextract.extract(domain1).domain, tldextract.extract(domain2).domain

                    edit_distance = levenshtein_distance(domain1, domain2)
                    normalized_edit_distance = edit_distance / max(len(domain1), len(domain2))

                    edit_distance_sub = levenshtein_distance(subdomain1, subdomain2)
                    if len(subdomain1) and len(subdomain2):
                        normalized_edit_distance_sub = edit_distance_sub / max(len(subdomain1), len(subdomain2))
                    else:
                        normalized_edit_distance_sub = 1

                    if normalized_edit_distance <= threshold or normalized_edit_distance_sub <= threshold and max(len(subdomain1), len(subdomain2))>3:
                        G.add_edge(i, j)

        # Use NetworkX's connected_components function to get the clusters
        clusters = list(nx.connected_components(G))

        return clusters

    @staticmethod
    def cluster_to_timeseries(cluster, all_dates):
        # Extract the dates from the cluster and count the number of screenshots for each date.
        dates = [screenshot_date for screenshot, screenshot_date, target in cluster]
        date_counts = Counter(dates)

        # Fill in any missing dates with zero counts.
        full_counts = [date_counts.get(date, 0) for date in all_dates]

        # Compute the cumulative counts.
        cumulative_counts = np.cumsum(full_counts)

        return all_dates, cumulative_counts

    def cluster_shot_representations(self, shot_path_list, target_list):
        n = len(shot_path_list)
        same_target_mask = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                same_target = target_list[i] == target_list[j]
                if same_target:
                    same_target_mask[i, j] = 1
                    same_target_mask[j, i] = 1  # The matrix is symmetric

        domain_name_list = [os.path.basename(os.path.dirname(x)) for x in shot_path_list]
        clusters = self.similarity_threshold_clustering(same_target_mask, domain_name_list, 0.5)

        clusters_path = []
        for it, cls in enumerate(clusters):
            shots_under_cls = [(shot_path_list[ind],
                                os.path.basename(os.path.dirname(os.path.dirname(shot_path_list[ind]))),
                                target_list[ind]) for ind in cls]
            clusters_path.append(shots_under_cls)
        return clusters_path

    def visualize_campaign(self, clusters):
        # Initialize Figure
        plt.figure(figsize=(15, 8))

        # Define Color Palette
        colors = cycle(sns.color_palette("husl", 9))

        # Extract Unique Dates and Sort Clusters
        all_dates = set()
        for cluster in clusters:
            _, dates, _ = zip(*cluster)
            all_dates.update(dates)
        all_dates = sorted(list(all_dates), key=lambda date: datetime.strptime(date, '%Y-%m-%d'))

        # Sort clusters by their earliest date
        sorted_clusters = sorted(clusters,
                                 key=lambda cluster: min(datetime.strptime(date, '%Y-%m-%d')
                                                         for _, date, _ in cluster))

        # Offset variable
        offset = 0
        offset_increment = 0.02  # Adjusted for clarity
        campaign_period = []

        # Plot Time Series for Each Cluster
        for i, cluster in enumerate(sorted_clusters):
            _, dates, targets = zip(*cluster)

            # Filter Clusters
            if len(cluster) < 4 or targets[0] in ['outlook.com', 'microsoft.com']:
                continue

            # Convert Cluster to Time Series
            dates, counts = self.cluster_to_timeseries(cluster, all_dates)

            # Add offset to counts
            offset_counts = [count + offset for count in counts]

            # Increment offset for next line
            offset += offset_increment

            # Identify Indices for First and Last Increases
            first_increase_index, last_increase_index = None, None
            for i in range(1, len(counts)):
                if counts[i] > counts[i - 1]:
                    last_increase_index = i
                    if first_increase_index is None:
                        first_increase_index = i

            # Trim and Plot Time Series
            if first_increase_index is not None and last_increase_index is not None:
                trimmed_dates = dates[first_increase_index:last_increase_index + 1]
                trimmed_counts = offset_counts[first_increase_index:last_increase_index + 1]
                color = next(colors)
                trimmed_indices = [all_dates.index(date) for date in trimmed_dates]
                plt.plot(trimmed_indices, trimmed_counts, marker='o', color=color,
                         label=f'Target = {targets[0]}',
                         linewidth=3, markersize=10)
                campaign_period.append(last_increase_index - first_increase_index + 1)

        print('Average campaign period: ', np.mean(campaign_period))

        # Configure Plot Aesthetics
        plt.xticks(range(len(all_dates)), all_dates, rotation=45, ha='right', fontsize=14)
        plt.yticks(np.arange(0, max(offset_counts) + 1, 1), fontsize=14)
        plt.ylim(bottom=0)
        plt.xlabel('Date', fontsize=18, color='black')
        plt.ylabel('Cumulative number of screenshots', fontsize=18, color='black')
        plt.legend(fontsize=20, loc='upper left', frameon=False)

        # Add Minimalist Grid Lines
        plt.grid(axis='x', linestyle='--', linewidth=0.5, color='gray')
        plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray')

        # Finalize and Save Plot
        plt.tight_layout()
        plt.savefig('./field_study/plots/campaign.png', dpi=300)  # Increase DPI for better quality
        plt.close()

if __name__ == '__main__':
    base = "./datasets/phishing_TP_examples"
    # '''geolocation'''
    gs_sheet = gwrapper_monitor()
    rows = gs_sheet.get_records()
    # geo_loc_list = list(map(lambda x: tuple(map(float, x['geo_loc'].split(','))), filter(lambda x: x['geo_loc'] != 0, rows)))
    # IPAnalysis.geoplot(geo_loc_list)
    #
    # '''phishing counts over time'''
    # start_date = date(2023, 8, 7)
    # today = datetime.today().date()
    # # end_date = today + timedelta(days=1)
    # end_date = today
    # dates = []
    # llm_counts = []
    # pedia_counts = []
    # intention_counts = []
    #
    # for single_date in daterange(start_date, end_date):
    #     date_ = single_date.strftime("%Y-%m-%d")
    #     llm_pos = get_pos_site('./field_study/results/{}_phishllm.txt'.format(date_))
    #     pedia_pos = get_pos_site('./field_study/results/{}_phishpedia.txt'.format(date_))
    #     intention_pos = get_pos_site('./field_study/results/{}_phishintention.txt'.format(date_))
    #     dates.append(date_)
    #     llm_counts.append(len(llm_pos))
    #     pedia_counts.append(len(pedia_pos))
    #     intention_counts.append(len(intention_pos))
    #
    # '''# of phishing over time'''
    # GeneralAnalysis.visualize_count(dates, llm_counts, pedia_counts, intention_counts)
    #
    # '''tld distribution domain age distribution'''
    # domains = list(map(lambda x: x['foldername'], rows))
    # domain_ages = list(map(lambda x: x['domain_age'], rows))
    # DomainAnalysis.tld_distribution(domains)
    # alexa_urls = [x.strip().split(',')[1] for x in open('./datasets/top-1m.csv').readlines()]
    # DomainAnalysis.tld_distribution(alexa_urls)
    # DomainAnalysis.domain_age_distribution(domain_ages)

    '''brand'''
    # df = pd.DataFrame(rows)
    # group_counts = df.groupby('brand').size()
    # sorted_groups = group_counts.sort_values(ascending=False)
    # print(sorted_groups)
    # top_group = sorted_groups.index[0]

    # Print the top group
    # print(f"Top group: {top_group}")
    # print(df[df['brand'] == top_group])

    # Sort groups by the frequency of the most common item in 'data'
    # sorted_groups = frequency.sort_values(ascending=False)

    brands = list(map(lambda x: x['brand'], rows))
    # brand_counts = Counter(brands)
    # print(len(rows))
    # BrandAnalysis.visualize_brands(brands)

    '''sector'''
    # sectors = list(map(lambda x: x['sector'], rows))
    # BrandAnalysis.visualize_sectors(sectors)

    '''phishing campaign'''
    shot_path_list = list(map(lambda x: os.path.join(base, x['date'], x['foldername'], 'shot.png'), rows))
    campaign = CampaignAnalysis()
    clusters_path = campaign.cluster_shot_representations(shot_path_list, brands)
    campaign.visualize_campaign(clusters_path)

    # print('Num of phishing using the Western Digital MyCloud service = ', np.sum(['remotewd.com' in x for x in domains]))

