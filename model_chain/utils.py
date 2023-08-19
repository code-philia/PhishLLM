from tldextract import tldextract
import numpy as np


def question_template_prediction(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, \n Question: A. This is a credential-requiring page. B. This is not a credential-requiring page. \n Answer: "
        }


def question_template_brand(html_text):
    return \
        {
            "role": "user",
            "content": f"Given the HTML webpage text: <start>{html_text}<end>, Question: What is the brand's domain? Answer: "
        }

def question_template_caption(logo_caption, logo_ocr):
    return \
        {
            "role": "user",
            "content": f"Given the following description on the brand's logo: <start>{logo_caption} {logo_ocr}<end>, Question: What is the brand's domain? Answer: "
        }


def compute_overlap_areas_between_lists(bboxes1, bboxes2):
    # Convert bboxes lists to 3D arrays
    bboxes1 = np.array(bboxes1)[:, np.newaxis, :]
    bboxes2 = np.array(bboxes2)

    # Compute overlap for x and y axes separately
    overlap_x = np.maximum(0, np.minimum(bboxes1[:, :, 2], bboxes2[:, 2]) - np.maximum(bboxes1[:, :, 0], bboxes2[:, 0]))
    overlap_y = np.maximum(0, np.minimum(bboxes1[:, :, 3], bboxes2[:, 3]) - np.maximum(bboxes1[:, :, 1], bboxes2[:, 1]))

    # Compute overlapping areas for each pair
    overlap_areas = overlap_x * overlap_y
    return overlap_areas


def expand_bbox(bbox, image_width, image_height, expand_ratio):
    # Extract the coordinates
    x1, y1, x2, y2 = bbox

    # Calculate the center
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate new width and height
    new_width = (x2 - x1) * expand_ratio
    new_height = (y2 - y1) * expand_ratio

    # Determine new coordinates
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Ensure coordinates are legitimate
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width, new_x2)
    new_y2 = min(image_height, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]
