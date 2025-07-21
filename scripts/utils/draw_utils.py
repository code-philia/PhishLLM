from PIL import Image, ImageDraw, ImageFont
from PIL import Image
import numpy as np
from typing import List
import cv2

def draw_annotated_image_nobox(image: Image.Image, txt: str):
    # Convert the image to RGBA for transparent overlay
    image = image.convert('RGBA')

    # Create a drawing context
    draw = ImageDraw.Draw(image)

    # Load a larger font for text annotations
    font = ImageFont.truetype(font="./utils/fonts/arialbd.ttf", size=25)

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
    font = ImageFont.truetype(font="./utils/fonts/arialbd.ttf", size=30)

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
    font = ImageFont.truetype(font="./utils/fonts/arialbd.ttf", size=18)
    text_width, text_height = draw.textsize(combined_text, font=font)

    # Create an image with extra space at the bottom for the concatenated text
    new_height = result.height + text_height + 10  # 10 for padding
    final_image = Image.new('RGBA', (result.width, new_height), (255, 255, 255, 255))
    final_image.paste(result, (0, 0))

    draw_final = ImageDraw.Draw(final_image)
    draw_final.text((10, result.height + 5), combined_text, font=font, fill="black")

    return final_image