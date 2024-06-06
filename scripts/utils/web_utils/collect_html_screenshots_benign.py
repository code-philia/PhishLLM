import os.path
import shutil
import time
import pandas as pd
from scripts.utils.web_utils.web_utils import is_uniform_color, cleanup_drivers, apply_custom_styles, init_driver_pool
from multiprocessing import Pool
from itertools import cycle

# Function to process domains
def process_domains(data):
    domain_list, root_dir = data
    drivers = init_driver_pool(size=2)  # Adjust size based on your resource capacity
    driver_cycle = cycle(drivers)  # Use itertools.cycle for round-robin usage of drivers

    for domain in domain_list:
        if len(os.listdir(root_dir)) >= 5000:
            cleanup_drivers(drivers)
            exit()
        driver = next(driver_cycle)
        target = f'https://{domain}'
        domain_path = os.path.join(root_dir, domain)
        if os.path.exists(os.path.join(domain_path, 'shot.png')):
            continue
        os.makedirs(domain_path, exist_ok=True)
        print(f'Target URL = {target}')

        try:
            driver.get(target)
            time.sleep(3)
            driver.scroll_to_top()
            apply_custom_styles(driver)

            # Save the page source and screenshot
            with open(os.path.join(domain_path, 'index.html'), 'w', encoding='utf-8') as f:
                f.write(driver.page_source())
            with open(os.path.join(domain_path, 'info.txt'), 'w', encoding='utf-8') as f:
                f.write(target)
            driver.save_screenshot(os.path.join(domain_path, 'shot.png'))

            # Remove domain directory if screenshot shows mostly uniform color
            if is_uniform_color(os.path.join(domain_path, 'shot.png')):
                shutil.rmtree(domain_path)
                print(f'{domain} - removed, mostly uniform color.')
            else:
                print(f'{domain} - valid screenshot saved.')
            # Assume further actions here
        except Exception as e:
            print(f'Error with {domain}: {e}')
            shutil.rmtree(domain_path)

    cleanup_drivers(drivers)


def main():
    root_dir = './datasets/alexa_middle_5k'
    os.makedirs(root_dir, exist_ok=True)
    popular_1m = pd.read_csv('./datasets/tranco-top-1m.csv', header=None)
    middle_domains = popular_1m.iloc[:, 1].tolist()[100000:105000]

    # Split the domain list into chunks for multiprocessing
    chunk_size = len(middle_domains) // os.cpu_count()
    domain_chunks = [middle_domains[i:i + chunk_size] for i in range(0, len(middle_domains), chunk_size)]

    # Start multiprocessing
    try:
        with Pool(os.cpu_count()) as pool:
            pool.map(process_domains, [(chunk, root_dir) for chunk in domain_chunks])
    except KeyboardInterrupt:
        print("Interrupted by user, cleaning up...")
    finally:
        print("Final cleanup if any")


if __name__ == "__main__":

    main()

