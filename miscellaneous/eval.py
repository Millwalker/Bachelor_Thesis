from collections import defaultdict

# Read ground truth data
with open("mot/2persons_left/gt/gt.txt", 'r') as gt_file:
    gt_data = gt_file.readlines()

# Read tracking results data
with open("result/2persons_left_thesis_f.txt", 'r') as track_file:
    track_data = track_file.readlines()

# Parse ground truth data into a dictionary
gt_dict = defaultdict(dict)
for line in gt_data:
    frame_id, obj_id, *info = line.strip().split(',')
    frame_id = int(frame_id)
    obj_id = int(obj_id)
    gt_dict[frame_id][obj_id] = info  # Store data in a dictionary organized by frame ID and object ID

# Parse tracking results data into a dictionary
track_dict = defaultdict(dict)
for line in track_data:
    frame_id, obj_id, *info = line.strip().split(',')
    frame_id = int(frame_id)
    obj_id = int(obj_id)
    track_dict[frame_id][obj_id] = info  # Store data in a dictionary organized by frame ID and object ID

# Initialize variables for metrics
num_misses = 0
num_switches = 0
num_false_positives = 0
num_matches = 0
total_distance = 0
total_matched = 0
num_matches_id = 0
num_false_positives_id = 0
num_misses_id = 0
threshold = 20

# Loop through frames
for frame_id, gt_objects in gt_dict.items():
    # Get corresponding tracking results for this frame
    track_objects = track_dict.get(frame_id, {})

    # Loop through ground truth objects in this frame
    for obj_id, gt_info in gt_objects.items():
        if obj_id in track_objects:
            track_info = track_objects[obj_id]

            # Calculate distance (assuming Euclidean distance between centroids)
            gt_centroid = (float(gt_info[0]), float(gt_info[1]))
            track_centroid = (float(track_info[0]), float(track_info[1]))
            distance = ((gt_centroid[0] - track_centroid[0])**2 + (gt_centroid[1] - track_centroid[1])**2)**0.5
            total_distance += distance
            total_matched += 1

            # Check if the match is acceptable based on distance threshold (for MOTA/MOTP)
            if distance <= threshold:
                num_matches += 1
            else:
                num_false_positives += 1

            # Check if the match is acceptable based on another threshold (for IDP/IDR/IDF1)
            if distance <= threshold:
                num_matches_id += 1
            else:
                num_false_positives_id += 1

        else:
            num_misses += 1
            num_misses_id += 1

# Calculate 'num_switches'
for frame_id, track_objects in track_dict.items():
    # Get corresponding ground truth objects for this frame
    gt_objects = gt_dict.get(frame_id, {})

    # Loop through tracked objects in this frame
    for obj_id, track_info in track_objects.items():
        if obj_id in gt_objects:
            gt_info = gt_objects[obj_id]

            # Calculate distance (assuming Euclidean distance between centroids)
            gt_centroid = (float(gt_info[0]), float(gt_info[1]))
            track_centroid = (float(track_info[0]), float(track_info[1]))
            distance = ((gt_centroid[0] - track_centroid[0])**2 + (gt_centroid[1] - track_centroid[1])**2)**0.5

            # Check if the match is not acceptable based on distance threshold
            if distance > threshold:
                num_switches += 1

# Calculate MOTA and MOTP
num_gt = sum(len(objects) for objects in gt_dict.values())
MOTA = 1 - (num_misses + num_false_positives + num_switches) / num_gt
img_width=1280
img_height=800
MOTP = total_distance / total_matched if total_matched > 0 else 0.0
max_possible_distance = ((img_width**2) + (img_height**2))**0.5
MOTP = 1 - (MOTP / max_possible_distance) if max_possible_distance > 0 else 0.0 #normalized


# Calculate IDP, IDR, and IDF1
IDP = num_matches_id / (num_matches_id + num_false_positives_id) if (num_matches_id + num_false_positives_id) > 0 else 0.0
IDR = num_matches_id / (num_matches_id + num_misses_id) if (num_matches_id + num_misses_id) > 0 else 0.0
IDF1 = 2 * (IDP * IDR) / (IDP + IDR) if (IDP + IDR) > 0 else 0.0

print(f"MOTA: {MOTA}")
print(f"MOTP: {MOTP}")
print(f"IDP: {IDP}")
print(f"IDR: {IDR}")
print(f"IDF1: {IDF1}")
print(f"Number of Switches: {num_switches}")
