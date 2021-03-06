import torch
import os
import re
from io import BytesIO
import IPython
import PIL
import copy
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import time


def boxes_iou(box1, box2):
    # Get the Width and Height of each bounding box
    width_box1 = box1[2]
    height_box1 = box1[3]
    width_box2 = box2[2]
    height_box2 = box2[3]

    # Calculate the area of the each bounding box
    area_box1 = width_box1 * height_box1
    area_box2 = width_box2 * height_box2

    # Find the vertical edges of the union of the two bounding boxes
    mx = min(box1[0] - width_box1 / 2.0, box2[0] - width_box2 / 2.0)
    Mx = max(box1[0] + width_box1 / 2.0, box2[0] + width_box2 / 2.0)

    # Calculate the width of the union of the two bounding boxes
    union_width = Mx - mx

    # Find the horizontal edges of the union of the two bounding boxes
    my = min(box1[1] - height_box1 / 2.0, box2[1] - height_box2 / 2.0)
    My = max(box1[1] + height_box1 / 2.0, box2[1] + height_box2 / 2.0)

    # Calculate the height of the union of the two bounding boxes
    union_height = My - my

    # Calculate the width and height of the area of intersection of the two bounding boxes
    intersection_width = width_box1 + width_box2 - union_width
    intersection_height = height_box1 + height_box2 - union_height

    # If the the boxes don't overlap then their IOU is zero
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0

    # Calculate the area of intersection of the two bounding boxes
    intersection_area = intersection_width * intersection_height

    # Calculate the area of the union of the two bounding boxes
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate the IOU
    iou = intersection_area / union_area

    return iou

def nms(boxes, iou_thresh):
    # If there are no bounding boxes do nothing
    if len(boxes) == 0:
        return boxes

    # Create a PyTorch Tensor to keep track of the detection confidence
    # of each predicted bounding box
    det_confs = torch.zeros(len(boxes))

    # Get the detection confidence of each predicted bounding box
    for i in range(len(boxes)):
        det_confs[i] = boxes[i][4]

    # Sort the indices of the bounding boxes by detection confidence value in descending order.
    # We ignore the first returned element since we are only interested in the sorted indices
    _, sortIds = torch.sort(det_confs, descending=True)

    # Create an empty list to hold the best bounding boxes after
    # Non-Maximal Suppression (NMS) is performed
    best_boxes = []

    # Perform Non-Maximal Suppression 
    for i in range(len(boxes)):

        # Get the bounding box with the highest detection confidence first
        box_i = boxes[sortIds[i]]

        # Check that the detection confidence is not zero
        if box_i[4] > 0:

            # Save the bounding box 
            best_boxes.append(box_i)

            # Go through the rest of the bounding boxes in the list and calculate their IOU with
            # respect to the previous selected box_i. 
            for j in range(i + 1, len(boxes)):
                box_j = boxes[sortIds[j]]

                # If the IOU of box_i and box_j is higher than the given IOU threshold set
                # box_j's detection confidence to zero. 
                if boxes_iou(box_i, box_j) > iou_thresh:
                    box_j[4] = 0

    return best_boxes


def detect_objects(model, img, iou_thresh, nms_thresh, device, verbose=False, return_time=False):
    # Set the model to evaluation mode.
    model.eval()

    # Convert the image from a NumPy ndarray to a PyTorch Tensor of the correct shape.
    # The image is transposed, then converted to a FloatTensor of dtype float32, then
    # Normalized to values between 0 and 1, and finally unsqueezed to have the correct
    # shape of 1 x 3 x 416 x 416
    start = time.time()
    img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

    img = img.to(device)

    # Feed the image to the neural network with the corresponding NMS threshold.
    # The first step in NMS is to remove all bounding boxes that have a very low
    # probability of detection. All predicted bounding boxes with a value less than
    # the given NMS threshold will be removed.
    list_boxes = model(img, nms_thresh)

    # Make a new list with all the bounding boxes returned by the neural network
    boxes = list_boxes[0][0] + list_boxes[1][0] + list_boxes[2][0]
    detection_time = time.time() - start

    # Perform the second step of NMS on the bounding boxes returned by the neural network.
    # In this step, we only keep the best bounding boxes by eliminating all the bounding boxes
    # whose IOU value is higher than the given IOU threshold
    start = time.time()
    boxes = nms(boxes, iou_thresh)
    suppression_time = time.time() - start


    # Print the time it took to detect objects
    if verbose:
        print('\n\nIt took {:.3f}'.format(detection_time + suppression_time), 'seconds to detect the objects in the image.\n')

    # Print the number of objects detected
    if verbose:
        print('Number of Objects Detected:', len(boxes), '\n')

    if return_time:
        return boxes, detection_time, suppression_time
    return boxes


def print_objects(boxes, class_names):
    print('Objects Found and Confidence Level:\n')
    for i in range(len(boxes)):
        box = boxes[i]
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            print('%i. %s: %f' % (i + 1, class_names[cls_id], cls_conf))


def plot_boxes(img_shape, img_axis, boxes, class_names, plot_labels, color=None):

    # Define a tensor used to set the colors of the bounding boxes
    colors = torch.FloatTensor([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    # Define a function to set the colors of the bounding boxes
    def get_color(c, x, max_val):
        ratio = float(x) / max_val * 5
        i = int(np.floor(ratio))
        j = int(np.ceil(ratio))

        ratio = ratio - i
        r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]

        return int(r * 255)

    # Get the width and height of the image
    width = img_shape[1]
    height = img_shape[0]

    # Plot the bounding boxes and corresponding labels on top of the image
    for i in range(len(boxes)):

        # Get the ith bounding box
        box = boxes[i]

        # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
        # of the bounding box relative to the size of the image. 
        x1 = int(np.around((box[0] - box[2] / 2.0) * width))
        y1 = int(np.around((box[1] - box[3] / 2.0) * height))
        x2 = int(np.around((box[0] + box[2] / 2.0) * width))
        y2 = int(np.around((box[1] + box[3] / 2.0) * height))

        # Set the default rgb value to red
        rgb = (1, 0, 0)

        # Use the same color to plot the bounding boxes of the same object class
        if len(box) >= 7 and class_names:
            cls_conf = box[5]
            cls_id = box[6]
            classes = len(class_names)
            offset = cls_id * 123457 % classes
            red = get_color(2, offset, classes) / 255
            green = get_color(1, offset, classes) / 255
            blue = get_color(0, offset, classes) / 255

            # If a color is given then set rgb to the given color instead
            if color is None:
                rgb = (red, green, blue)
            else:
                rgb = color

        # Calculate the width and height of the bounding box relative to the size of the image.
        width_x = x2 - x1
        width_y = y1 - y2

        # Set the postion and size of the bounding box. (x1, y2) is the pixel coordinate of the
        # lower-left corner of the bounding box relative to the size of the image.
        rect = patches.Rectangle((x1, y2),
                                 width_x, width_y,
                                 linewidth=2,
                                 edgecolor=rgb,
                                 facecolor='none')

        # Draw the bounding box on top of the image
        img_axis.add_patch(rect)

        # If plot_labels = True then plot the corresponding label
        if plot_labels:
            # Create a string with the object class name and the corresponding object class probability
            conf_tx = class_names[cls_id] + ': {%.2f}' % (cls_conf)

            # Define x and y offsets for the labels
            lxc = (img_shape[1] * 0.266) / 100
            lyc = (img_shape[0] * 1.180) / 100

            # Draw the labels on top of the image
            img_axis.text(x1 + lxc, y1 - lyc, conf_tx, fontsize=10, color='k',

                          bbox=dict(facecolor=rgb, edgecolor=rgb, alpha=0.8))

    plt.show()


def show_array(a, fmt='jpeg'):
    """
    This function is to speed up the display of images in CoLab.
    Use 'jpeg' instead of 'png' (~5 times faster)
    :param a: Image as an array
    :param fmt: Image saved as of format
    :return: Nothing
    """
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))


def get_color(c, x, max_val):
    """
    A function to set the colors of the bounding boxes
    :param c:
    :param x:
    :param max_val:
    :return:
    """
    # Define a tensor used to set the colors of the bounding boxes
    colors = np.array([[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]])

    ratio = float(x) / max_val * 5
    i = int(np.floor(ratio))
    j = int(np.ceil(ratio))

    ratio = ratio - i
    r = (1 - ratio) * colors[i][c] + ratio * colors[j][c]

    return int(r * 255)


def cv2_put_text(img, text, text_offset_x, text_offset_y, background_color=(255, 255, 255), text_color=(255, 255, 255)):
    """
    A Function to write text on an image using openCV
    :param img: The image to write text on it
    :param text: The text to be written
    :param text_offset_x: The text bbox upper left point abscissa
    :param text_offset_y: The text bbox upper left point ordinate
    :param background_color: The text bbox background color
    :param text_color: The text color
    :return: Nothing
    """
    font_scale = 0.35
    font = cv2.FONT_HERSHEY_SIMPLEX

    # get the width and height of the text box
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    # make the coords of the box with a small padding of two pixels
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
    cv2.rectangle(img, box_coords[0], box_coords[1], background_color, cv2.FILLED)
    cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=text_color, thickness=1)


def annotate_frame_with_objects(original_frame, objects_bboxes, class_names, only_classes=None, confidence_threshold= 0.6, plot_labels=True, plot_class_confidence=False, text_color=(0,0,0)):
    """
    This function plots detected objects bounding boxes over images with class name and accuracy
    :param original_frame: A Frame(Image) from video
    :param objects_bboxes: Detected Objects Bounding boxes (output of yolo object detection model) and their class
    :param class_names: Array of class names
    :param only_classes: A list of class names to consider, if none consider all
    :param confidence_threshold:
    :param plot_labels: Whether to write down class label over bounding boxes or not
    :param plot_class_confidence: Whether to write down class confidence over bounding boxes or not
    :return: Masked Frame
    """
    masked_frame = copy.copy(original_frame)

    for i in range(len(objects_bboxes)):

        # Get the ith bounding box
        box = objects_bboxes[i]

        cls_id = box[6]
        cls_conf = box[5]

        if only_classes and not class_names[cls_id] in only_classes:
            continue

        if cls_conf<confidence_threshold:
            continue

        # Get the (x,y) pixel coordinates of the lower-left and lower-right corners
        # of the bounding box relative to the size of the image.
        x1 = int(np.around((box[0] - box[2] / 2.0) * masked_frame.shape[1]))
        y1 = int(np.around((box[1] - box[3] / 2.0) * masked_frame.shape[0]))
        x2 = int(np.around((box[0] + box[2] / 2.0) * masked_frame.shape[1]))
        y2 = int(np.around((box[1] + box[3] / 2.0) * masked_frame.shape[0]))

        # Calculate the width and height of the bounding box relative to the size of the image.
        width_x = x2 - x1
        width_y = y1 - y2

        # get color
        classes = len(class_names)
        offset = cls_id * 123457 % classes
        r = get_color(2, offset, classes)
        g = get_color(1, offset, classes)
        b = get_color(0, offset, classes)

        cv2.rectangle(masked_frame, (x1, y2), (x1 + width_x, y2 + width_y), (b, g, r), 1)

        if plot_labels:
            # Define x and y offsets for the labels
            lxc = (masked_frame.shape[1] * 0.266) / 100
            lyc = (masked_frame.shape[0] * 1.180) / 100

            # Plot class name
            cv2_put_text(masked_frame, class_names[cls_id], int(x1), int(y1)-1, background_color=(b, g, r), text_color=text_color)

        if plot_class_confidence:
            # Plot probability
            cv2_put_text(masked_frame, "{0:.2f}".format(cls_conf), int(x1), int(y2), background_color=(b, g, r), text_color=text_color)

    return masked_frame

def generate_yolo_train_test_files(images_dir, output_dir, classes, train_valid_split=0.8):
    train_output = output_dir+"/train.txt"
    valid_output = output_dir+"/valid.txt"
    data_output = output_dir+"/obj.data"
    names_output = output_dir+"/obj.names"
    backup_path = output_dir+"/backup"

    images = [ f for f in os.listdir(images_dir) if re.match(".*.jpg$",f)]
    train = np.random.choice(images, size=int(len(images)*train_valid_split) )
    valid = set(images) - set(train)

    # Write train.txt
    f_train = open(train_output, "w")
    for image_name in train:
        f_train.write(images_dir+"/"+image_name+"\n")
    f_train.close()

    # Write valid.txt
    f_valid = open(valid_output, "w")
    for image_name in valid:
        f_valid.write(images_dir+"/"+image_name+"\n")
    f_train.close()

    # create backup
    os.makedirs(backup_path)

    # Write obj.names
    f_names = open(names_output, "w")
    for class_name in classes:
        f_names.write(class_name+"\n")
    f_names.close()

    f_data = open(data_output, "w")
    f_data.write("classes="+str(len(classes))+"\n")
    f_data.write("train="+train_output+"\n")
    f_data.write("valid="+valid_output+"\n")
    f_data.write("names="+names_output+"\n")
    f_data.write("backup="+backup_path+"\n")
    f_data.close()


def replace_class_yolo_format(original_class, replace_class, images_labels_dir, image_label_file_regex=".*.txt$"):
    for file in os.listdir(images_labels_dir):
        if re.match(pattern=image_label_file_regex, string=file):
            # open file
            f = open(os.path.join(images_labels_dir, file), "r")
            file_content = f.read().split("\n")[:-1]
            f.close()
            file_output_content = []
            for line in file_content:
                bbox = line.split(" ")
                if int(bbox[0]) == original_class:
                    bbox[0] = str(replace_class)
                file_output_content.append(" ".join(bbox))
            f = open(os.path.join(images_labels_dir, file), "w")
            f.write("\n".join(file_output_content))
            f.close()
        else:
            continue