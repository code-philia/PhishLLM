import os.path
import shutil
import time
import tldextract.tldextract
from tqdm import tqdm
from scripts.utils.web_utils import CustomWebDriver
import json
import requests
import subprocess
from datetime import date
import numpy as np
from PIL import Image
from itertools import cycle
from multiprocessing import Pool, current_process
from selenium.common.exceptions import WebDriverException

def fetch_phish_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.text.split('\n')  # Split by newlines to get a list
        return data
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return []


def is_uniform_color(image_path, threshold=0.01, color_coverage=0.9):
    np_image = np.array(Image.open(image_path))

    # Calculate the standard deviation of all color channels
    std_dev = np.std(np_image, axis=(0, 1))

    # Check if the standard deviation across all color channels is below the threshold
    if np.all(std_dev < threshold):
        return True

    total_pixels = np_image.size / np_image.shape[2] if len(np_image.shape) == 3 else np_image.size

    # Flatten the array to 1D and convert to tuples if it's a color image
    if len(np_image.shape) == 3:
        np_image = np_image.reshape(-1, np_image.shape[2])
        np_image = [tuple(pixel) for pixel in np_image]
    else:
        np_image = np_image.flatten()

    # Count the most common color
    colors, counts = np.unique(np_image, axis=0, return_counts=True)
    most_common_color_count = np.max(counts)

    # Check if the most common color is at least 90% of the image
    if most_common_color_count / total_pixels >= color_coverage:
        return True

    return False


# Function to initialize a pool of WebDrivers
def init_driver_pool(size=2):
    drivers = []
    for _ in range(size):
        driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")
        driver.set_script_timeout(10)
        driver.set_page_load_timeout(30)
        drivers.append(driver)
    return drivers

# Clean up drivers
def cleanup_drivers(drivers):
    for driver in drivers:
        driver.quit()

def restart_driver(driver_index, drivers):
    """Restart a driver at a specific index in the drivers list."""
    drivers[driver_index].quit()
    drivers[driver_index] = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")
    drivers[driver_index].set_script_timeout(10)
    drivers[driver_index].set_page_load_timeout(30)


def apply_custom_styles(driver):
    """Inject CSS to modify page styles."""
    css_script = """
        var style = document.createElement('style');
        document.head.appendChild(style);
        style.appendChild(document.createTextNode(`
            body, body * { color: black !important; }
            ::-webkit-input-placeholder { color: black !important; }
            ::-moz-placeholder { opacity: 1; color: black !important; }
            :-ms-input-placeholder { color: black !important; }
            ::placeholder { color: black !important; }
        `));
    """
    driver.execute_script(css_script)

# Function to process domains
def process_domains(data):
    url_list, root_dir = data
    drivers = init_driver_pool(size=2)  # Adjust size based on your resource capacity
    driver_indices = cycle(range(len(drivers)))  # Track indices for restarting drivers

    for url in url_list:
        driver_index = next(driver_indices)
        driver = drivers[driver_index]
        domain = '.'.join(part for part in tldextract.extract(url) if part)
        domain_path = os.path.join(root_dir, domain)
        if os.path.exists(os.path.join(domain_path, 'shot.png')):
            continue
        os.makedirs(domain_path, exist_ok=True)

        print(f'Target URL = {url}')

        try:
            driver.get(url)
            time.sleep(3)
            driver.scroll_to_top()
            apply_custom_styles(driver)

            # Save the page source and screenshot
            with open(os.path.join(domain_path, 'index.html'), 'w', encoding='utf-8') as f:
                f.write(driver.page_source())
            with open(os.path.join(domain_path, 'info.txt'), 'w', encoding='utf-8') as f:
                f.write(url)
            driver.save_screenshot(os.path.join(domain_path, 'shot.png'))

            # Remove domain directory if screenshot shows mostly uniform color
            if is_uniform_color(os.path.join(domain_path, 'shot.png')):
                shutil.rmtree(domain_path)
                print(f'{domain} - removed, mostly uniform color.')
            else:
                print(f'{domain} - valid screenshot saved.')
            # Assume further actions here
        except WebDriverException as e:
            print(f'WebDriver error with {domain}: {e}, restarting driver.')
            restart_driver(driver_index, drivers)
            shutil.rmtree(domain_path)
        except Exception as e:
            print(f'Error with {domain}: {e}')
            shutil.rmtree(domain_path)

    cleanup_drivers(drivers)

def read_links(phish_list, path):
    phish_list = []
    if os.path.exists(path):
        phish_list_new = [x.strip() for x in open(path).readlines()]
        phish_list.extend(phish_list_new)
        print(f'From github repo {len(phish_list_new)}')
    return phish_list

def read_domains(phish_list, path):
    if not os.path.exists(path):
        return []
    phish_list_new = ['http://' + x.strip() for x in open(path).readlines()]
    phish_list.extend(phish_list_new)
    print(f'From github repo {len(phish_list_new)}')
    return phish_list

def main(root_dir):

    # Source 1: OpenPhish
    phish_list = fetch_phish_data("https://openphish.com/feed.txt")
    print(f'From openphish {len(phish_list)}')

    # Source 2: Public phishing feeds
    subprocess.run(["chmod", "+x", "./crawling_scripts/download_github_phishing_feed.sh"])
    subprocess.run(["./crawling_scripts/download_github_phishing_feed.sh"])

    for path in ['./datasets/phishing-links-ACTIVE-TODAY.txt', './datasets/phishing-links-ACTIVE-NOW.txt',
                 './datasets/phishing-links-ACTIVE.txt', './datasets/phishing-links-NEW-last-hour.txt',
                 './datasets/phishing-links-NEW-today.txt']:
        phish_list = read_links(phish_list, path)

    for path in ['./datasets/phishing-domains-NEW-today.txt', './datasets/phishing-domains-NEW-last-hour.txt',
                 './datasets/phishing-domains-ACTIVE.txt']:
        phish_list = read_domains(phish_list, path)

    # Remove duplicates
    phish_list = set(phish_list)
    if os.path.exists('./datasets/public_phishing_logged.txt'):
        existing = [x.strip() for x in open('./datasets/public_phishing_logged.txt').readlines()]
        phish_list = phish_list - set(existing)
    phish_list = [x for x in list(phish_list) if x]

    # Split the domain list into chunks for multiprocessing
    chunk_size = len(phish_list) // os.cpu_count()
    url_chunks = [phish_list[i:i + chunk_size] for i in range(0, len(phish_list), chunk_size)]

    # Start multiprocessing
    try:
        with Pool(os.cpu_count()) as pool:
            pool.map(process_domains, [(chunk, root_dir) for chunk in url_chunks])
    except KeyboardInterrupt:
        print("Interrupted by user, cleaning up...")
    finally:
        print("Final cleanup if any")

if __name__ == "__main__":

    '''Crawl domain alias'''
    # with open('./datasets/domain_alias.json', 'r') as f:
    #     domain_alias_dict = json.load(f)
    # os.makedirs('./datasets/domain_alias_100', exist_ok=True)
    #
    # for brand in list(domain_alias_dict.keys()):
    #     os.makedirs(os.path.join('./datasets/domain_alias_100', brand), exist_ok=True)
    #     domains = domain_alias_dict[brand]
    #     for d in domains:
    #         target = 'https://{}'.format(d)
    #         if os.path.exists(os.path.join('./datasets/domain_alias_100', brand, d, 'shot.png')):
    #             continue
    #         else:
    #             os.makedirs(os.path.join('./datasets/domain_alias_100', brand, d), exist_ok=True)
    #         print('Target URL = {}'.format(target))
    #
    #         try:
    #             driver.get(target)
    #             time.sleep(sleep_time)
    #         except Exception as e:
    #             print('Error {} when getting the URL, exit..'.format(e))
    #             driver.quit()
    #             driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")  # Using the proxy_url variable
    #             driver.set_script_timeout(timeout_time / 2)
    #             driver.set_page_load_timeout(timeout_time)
    #             continue
    #
    #         try:
    #             with open(os.path.join('./datasets/domain_alias_100', brand, d, 'index.html'), "w", encoding='utf-8') as f:
    #                 f.write(driver.page_source())
    #             driver.save_screenshot(os.path.join('./datasets/domain_alias_100', brand, d, 'shot.png'))
    #             print(brand, d)
    #         except Exception as e:
    #             print('Error {} when saving page source, exit..'.format(e))
    #
    #
    # driver.quit()

    '''OpenPhish Feed?'''
    today = date.today()
    formatted_date = today.strftime('%Y-%m-%d')
    root_dir = os.path.join('./datasets/public_phishing_feeds', formatted_date)
    os.makedirs(root_dir, exist_ok=True)

    main(root_dir)

    # driver.delete_all_cookies()
    # driver.quit()
    # driver = CustomWebDriver.boot(proxy_server="http://127.0.0.1:7890")  # Using the proxy_url variable
    # driver.set_script_timeout(timeout_time / 2)
    # driver.set_page_load_timeout(timeout_time)
    #
    # for date in os.listdir(root_dir):
    #     exists_domain = []
    #     for folder in os.listdir(os.path.join(root_dir, date)):
    #
    #         if len(os.listdir(os.path.join(root_dir, date, folder))) == 0:
    #             shutil.rmtree(os.path.join(root_dir, date, folder))
    #         if not os.path.exists(os.path.join(root_dir, date, folder, 'shot.png')):
    #             shutil.rmtree(os.path.join(root_dir, date, folder))
    #
    #         if is_uniform_color(os.path.join(root_dir, date, folder, 'shot.png')):
    #             shutil.rmtree(os.path.join(root_dir, date, folder))
    #         if folder in exists_domain:
    #             shutil.rmtree(os.path.join(root_dir, date, folder))
    #
    #         exists_domain.append(folder)
