# diagram_detector/post_processing.py

import torch
import torchvision
import numpy as np
from PIL import Image
import cv2
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from models.faster_rcnn_model import get_model  
import argparse

def detect_objects(model, img_tensor, device, conf_threshold=0.7):
    """
    Performs object detection on the input image tensor.

    Args:
        model: Trained object detection model.
        img_tensor: Input image tensor.
        device: Computation device (CPU or GPU).
        conf_threshold: Confidence threshold for detections.

    Returns:
        detections: Dictionary containing detected shapes, texts, and arrows.
    """
    model.eval()
    with torch.no_grad():
        predictions = model([img_tensor.to(device)])[0]

    shapes = []
    texts = []
    arrows = []

    # Mapping from label indices to category names 
    label_to_category_name = {
        1: 'circle',
        2: 'rectangle',
        3: 'diamond',
        4: 'oval',
        5: 'arrow',
        6: 'text',
    }

    shape_labels = {1, 2, 3, 4}  # Labels corresponding to shapes
    text_label = 6               # Label corresponding to text
    arrow_label = 5              # Label corresponding to arrows
    shape_idx = 0
    text_idx = 0
    arrow_idx = 0

    for idx in range(len(predictions['boxes'])):
        label = predictions['labels'][idx].item()
        box = predictions['boxes'][idx].cpu().numpy()
        score = predictions['scores'][idx].item()

        if score < conf_threshold:
            continue

        detection = {
            'box': box,
            'label': label,
            'score': score,
            'category_name': label_to_category_name.get(label, 'Unknown'),
            'idx': 0
        }

        if label in shape_labels:
            detection['idx'] = shape_idx
            shapes.append(detection)
            shape_idx += 1
        elif label == text_label:
            detection['idx'] = text_idx
            texts.append(detection)
            text_idx += 1
        elif label == arrow_label:
            detection['idx'] = arrow_idx
            arrows.append(detection)
            arrow_idx += 1

    detections = {
        'shapes': shapes,
        'texts': texts,
        'arrows': arrows
    }

    return detections

# def extract_arrow_endpoints(arrow_image):
#     """
#     Extracts the endpoints of an arrow using skeletonization.

#     Args:
#         arrow_image: Cropped image containing the arrow.

#     Returns:
#         endpoints: List of (x, y) coordinates of arrow endpoints.
#     """
#     # Convert to grayscale
#     gray = cv2.cvtColor(arrow_image, cv2.COLOR_BGR2GRAY)
#     # Thresholding
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#     # Skeletonization
#     skeleton = skeletonize(binary // 255)
#     # Find endpoints
#     endpoints = find_skeleton_endpoints(skeleton)
#     return endpoints

# def find_skeleton_endpoints(skeleton):
#     """
#     Finds endpoints in a skeletonized image.

#     Args:
#         skeleton: Binary skeletonized image.

#     Returns:
#         endpoints: List of (x, y) coordinates of endpoints.
#     """
#     # Define the kernel to detect endpoints
#     kernel = np.array([[1, 1, 1],
#                        [1, 10, 1],
#                        [1, 1, 1]])
#     # Convolve the kernel with the skeleton image
#     filtered = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
#     # Endpoints have a value of 11 after filtering
#     endpoints = np.argwhere(filtered == 11)
#     # Convert to (x, y) format
#     endpoints = [(int(point[1]), int(point[0])) for point in endpoints]
#     # plt.figure(figsize=(6, 6))
#     # plt.imshow(skeleton, cmap='gray')
#     # plt.scatter([pt[0] for pt in endpoints], [pt[1] for pt in endpoints], color='red', label='Endpoints')
#     # plt.title('Skeleton with Detected Endpoints')
#     # plt.legend()
#     # plt.axis('off')
#     # plt.savefig('../endpoint_plot_1.png')
#     # plt.show()
#     return endpoints


def extract_arrow_endpoints_simple(arrow_box):
    """
    Compute arrow endpoints based on the arrow bounding box.

    Args:
        arrow_box: [xmin, ymin, xmax, ymax] coordinates of the arrow's bounding box.
    
    Returns:
        endpoints: List of two (x, y) tuples representing the arrow's endpoints.
    """
    xmin, ymin, xmax, ymax = arrow_box
    width = xmax - xmin
    height = ymax - ymin

    if width > height:
        # Horizontal arrow
        mid_y = (ymin + ymax) / 2.0
        left_endpoint = (xmin, mid_y)
        right_endpoint = (xmax, mid_y)
        return [left_endpoint, right_endpoint]
    else:
        # Vertical arrow
        mid_x = (xmin + xmax) / 2.0
        top_endpoint = (mid_x, ymin)
        bottom_endpoint = (mid_x, ymax)
        return [top_endpoint, bottom_endpoint]


def find_nearest_shape(point, shapes, max_distance=1000):
    """
    Finds the nearest shape to a given point.

    Steps:
    1. Check if the point is inside any shape's bounding box.
       If yes, return that shape immediately.
    2. If not inside a bounding box, find the shape whose center is closest
       to the point and ensure that this distance is less than max_distance.

    Args:
        point: (x, y) coordinates of the point.
        shapes: List of detected shapes.
        max_distance: Maximum distance to consider for association.

    Returns:
        nearest_shape: The nearest shape dictionary or None if not found.
    """
    x, y = point
    inside_candidates = []
    center_candidates = []

    for shape in shapes:
        box = shape['box']
        xmin, ymin, xmax, ymax = box

        # Check if inside bounding box
        if xmin <= x <= xmax and ymin <= y <= ymax:
            # Perfect match
            return shape

        # Compute distance to shape center
        shape_center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
        distance = ((x - shape_center[0])**2 + (y - shape_center[1])**2)**0.5

        center_candidates.append((distance, shape))

    # No shape found by inside check, fallback to nearest center
    center_candidates.sort(key=lambda c: c[0])
    if center_candidates:
        dist, shape = center_candidates[0]
        if dist <= max_distance:
            return shape

    return None


def crop_image(image, box):
    """
    Crops a region from the image based on the bounding box.

    Args:
        image: Original image (numpy array).
        box: Bounding box [xmin, ymin, xmax, ymax].

    Returns:
        Cropped image region.
    """
    xmin, ymin, xmax, ymax = box.astype(int)
    return image[ymin:ymax, xmin:xmax]


def build_relationships(img_cv, detections, text_only=False):
    """
    Builds relationships between shapes based on arrows, using a simpler arrow endpoint extraction.
    
    Args:
        img_cv: Original image in OpenCV (BGR) - not really needed now since we don't skeletonize.
        detections: Dictionary containing detected shapes, texts, and arrows.

    Returns:
        graph: List of connections representing the diagram graph.
    """
    if text_only:
        shapes = detections['texts']
    else:
        shapes = detections['shapes']
    arrows = detections['arrows']
    graph = []

    print(f"Number of shapes: {len(shapes)}, Number of arrows: {len(arrows)}")

    for arrow in arrows:
        print(arrow)
        arrow_box = arrow['box']
        endpoints = extract_arrow_endpoints_simple(arrow_box)
        
        print(f"\nProcessing arrow at box {arrow_box}, endpoints found: {endpoints}")

        connections = []
        for endpoint in endpoints:
            connected_shape = find_nearest_shape(endpoint, shapes, max_distance=1000)
            if connected_shape is not None:
                connections.append(connected_shape)
                print(f"Endpoint {endpoint} connected to shape {connected_shape['category_name']} at {connected_shape['idx']}")
            else:
                print(f"Endpoint {endpoint} found no shape within 1000px.")
                connections = []
                break

        if len(connections) == 2:
            # Ensure distinct shapes
            if connections[0] is not connections[1]:
                source_shape, target_shape = connections
                print(f"Arrow connects {source_shape['category_name']}{source_shape['idx']} to {target_shape['category_name']}{target_shape['idx']}")
                graph.append({
                    'source': source_shape,
                    'target': target_shape,
                    'arrow': arrow
                })
            else:
                print("Both endpoints connect to the same shape; no connection formed.")
        else:
            print("Could not form a connection for this arrow.")

    return graph


def visualize_relationships(img_cv, detections, graph, text_only=False):
    """
    Visualizes the detections and relationships on the image.

    Args:
        img_cv: Original image in OpenCV format (BGR).
        detections: Dictionary containing detected shapes, texts, and arrows.
        graph: List of connections representing the diagram graph.
    """
    # Convert BGR to RGB for visualization
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)

    # Draw shapes
    if not text_only:
        for shape in detections['shapes']:
            box = shape['box']
            xmin, ymin, xmax, ymax = box
            width = xmax - xmin
            height = ymax - ymin
            rect = patches.Rectangle(
                (xmin, ymin),
                width,
                height,
                linewidth=2,
                edgecolor='blue',
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                xmin,
                ymin - 5,
                f"{shape['category_name']} {shape['idx']}",
                color='blue',
                fontsize=10,
                backgroundcolor='white'
            )

    for shape in detections['texts']:
        box = shape['box']
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor='blue',
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            xmin,
            ymin - 5,
            f"{shape['category_name']} {shape['idx']}",
            color='blue',
            fontsize=10,
            backgroundcolor='white'
        )

    # Draw arrows
    for arrow in detections['arrows']:
        box = arrow['box']
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle(
            (xmin, ymin),
            width,
            height,
            linewidth=2,
            edgecolor='green',
            facecolor='none'
        )
        ax.add_patch(rect)

    # Draw relationships
    for edge in graph:
        source_box = edge['source']['box']
        target_box = edge['target']['box']
        # Draw an arrow from source to target
        source_center = ((source_box[0] + source_box[2]) / 2, (source_box[1] + source_box[3]) / 2)
        target_center = ((target_box[0] + target_box[2]) / 2, (target_box[1] + target_box[3]) / 2)
        ax.annotate("",
                    xy=target_center,
                    xytext=source_center,
                    arrowprops=dict(arrowstyle="->", color='red', linewidth=5))

    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../relationship_detect_2.png')
    plt.show()

def main():
    # Load the image
    parser = argparse.ArgumentParser(description="Process an image and determine arrow-based relationships between diagram elements.")
    parser.add_argument('-f', '--file', required=False, help='Path to the image to be processed')
    args = parser.parse_args()
    if args.file:
        image_path = args.file
    else:
        image_path = 'FC_B/test/writer023_fc_019.png' 
    #image_path = 'FC_B/train/writer000_fc_021.png'
    #image_path = '../IMG_0015.jpg'
    img_pil = Image.open(image_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img_tensor = torchvision.transforms.functional.to_tensor(img_pil)

    # Load the trained model
    num_classes = 7  # Number of classes including background
    model = get_model(num_classes)
    model.load_state_dict(torch.load('models/saved_models/gen_detector.pth'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Detect objects
    detections = detect_objects(model, img_tensor, device, conf_threshold=0.7)

    # Build relationships
    graph = build_relationships(img_cv, detections, text_only=False)

    # Output the graph data
    for edge in graph:
        source_label = edge['source']['category_name']
        target_label = edge['target']['category_name']
        source_idx = edge['source']['idx']
        target_idx = edge['target']['idx']
        print(f"{source_label} {source_idx} -> {target_label} {target_idx}")

    # Visualize relationships
    visualize_relationships(img_cv, detections, graph, text_only=False)


if __name__ == '__main__':
    main()
