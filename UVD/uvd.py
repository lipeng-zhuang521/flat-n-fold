'''
ground truth
metric
'''

import os
import pandas as pd
import numpy as np
import glob

base_dir = '/media/ubb/T9/useful_data'  # Adjust this path to your specific directory

def process_csv_files(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('output.csv'):
            # if file.startswith('output') and file.endswith('process.csv'):
                # Construct full file path
                file_path = os.path.join(subdir, file)
                # Load CSV into DataFrame
                df = pd.read_csv(file_path)

                # Transform the DataFrame
                new_df = df['Directory'].str.extract(r'(?P<Garment>^[^/]+)/(?P<Camera>[^/]+)')
                new_df['Participant'] = os.path.basename(subdir) 
                new_df['Index'] = df['Index']

                # Rearrange columns
                new_df = new_df[['Participant', 'Garment', 'Camera', 'Index']]
                # file_path_new = file_path.replace('output', 'output_new')
                # Save the transformed DataFrame
                new_df.to_csv(file_path, index=False)
                print(f"Processed and saved: {file_path}")

# Run the processing function
# process_csv_files(base_dir)

# extract groudtruth
'''
# garment on the table phrases: find the two hand place timestep, we only care afterwards
# from 0 to 1: pick from 1 to 0:place
# find two hands every from 0 to 1 and from 1 to 0.
# if two hand cross happen, do not use this data
# then output the index for both hand (keep one timestep if two hand same)
'''

import ast
# Load the CSV file


# Define function to track picks and places
def track_actions(df, gripper):
    changes = df[f'{gripper}_gripper'].diff()
    picks = df[changes == 1]['time']
    places = df[changes == -1]['time']
    return picks, places



def pair_actions(picks, places):
    paired_actions = []
    place_iter = iter(places)
    try:
        current_place = next(place_iter)
        for pick in picks:
            while pick > current_place:  # Skip places before the current pick
                current_place = next(place_iter)
            paired_actions.append((pick, current_place))
    except StopIteration:
        pass
    return paired_actions
def merge_actions(all_actions):
    merged_actions = []
    i = 0
    while i < len(all_actions):
        current_time, current_action = all_actions[i]
        # Check if the next action exists and occurs at the same time
        if i + 1 < len(all_actions) and all_actions[i + 1][0] == current_time:
            # If actions from both hands occur at the same time, merge them
            next_time, next_action = all_actions[i + 1]
            if ('pick' in current_action and 'pick' in next_action) or ('place' in current_action and 'place' in next_action):
                if 'pick' in current_action:
                    merged_actions.append((current_time, 'total_pick'))
                else:
                    merged_actions.append((current_time, 'total_place'))
                i += 1  # Skip the next action as it has been merged
        else:
            # If there is no overlap, add the current action
            merged_actions.append((current_time, current_action))
        i += 1
    return merged_actions




# final_actions = [(1717781420389875437, 'right_pick'), (1717781421599617143, 'right_place'), 
#                  (1717781423820140682, 'left_pick'), (1717781426291674731, 'left_place'), 
#                  (1717781429183325392, 'right_pick'), (1717781430742940108, 'left_pick'), 
#                  (1717781432741649671, 'left_place'), (1717781434621903921, 'right_place'), 
#                  (1717781436973407997, 'total_pick'), (1717781439061062340, 'total_place'), 
#                  (1717781440602049984, 'right_pick'), (1717781442214709890, 'right_place'), 
#                  (1717781444708076008, 'total_pick'), (1717781446308085171, 'total_place')]

def classify_actions(actions):
    normal_situations = []
    special_situations = []
    other_situations = []

    i = 0
    while i < len(actions) - 1:
        # Check pairs for normal situations
        current_time, current_action = actions[i]
        next_time, next_action = actions[i + 1]

        if (current_action.endswith('pick') and next_action.endswith('place') and
            current_action.split('_')[0] == next_action.split('_')[0]):
            # Normal situation
            normal_situations.append((current_time, current_action, next_time, next_action))
            i += 2  # Move to the next pair
        else:
            # We need at least two more actions to check for a special situation
            if i < len(actions) - 2:
                second_time, second_action = actions[i + 1]
                third_time, third_action = actions[i + 2]

                # Determine if there's a 'total' involved in the first three actions
                if 'total' in current_action or 'total' in second_action or 'total' in third_action:
                    # Handling sequences with 'total' that only require three actions
                    if (current_action.endswith('pick') and
                        second_action.endswith('pick') and
                        third_action.endswith('place')):
                        special_situations.append((current_time, current_action, third_time, third_action))
                        i += 3  # Move past these three actions
                    elif (current_action.endswith('pick') and
                        second_action.endswith('place') and
                        third_action.endswith('place')):
                        special_situations.append((current_time, current_action, third_time, third_action))
                        i += 3  # Move past these three actions
                    else:
                        other_situations.append((current_time, current_action))
                        i += 1
                else:
                    # Check if there's a fourth action to complete a four-action sequence
                    if i < len(actions) - 3:
                        fourth_time, fourth_action = actions[i + 3]
                        picks = (current_action, second_action)
                        places = (third_action, fourth_action)

                        # Analyze the pattern of picks and places
                        if picks == ('left_pick', 'right_pick'):
                            if places == ('left_place', 'right_place'):
                                special_situations.append((current_time, current_action, fourth_time, fourth_action))
                            elif places == ('right_place', 'left_place'):
                                special_situations.append((current_time, current_action, fourth_time, fourth_action))
                        elif picks == ('right_pick', 'left_pick'):
                            if places == ('left_place', 'right_place'):
                                special_situations.append((second_time, second_action, fourth_time, fourth_action))
                            elif places == ('right_place', 'left_place'):
                                special_situations.append((second_time, second_action, fourth_time, fourth_action))
                        
                        i += 4  # Move past these four actions
                    else:
                        other_situations.append((current_time, current_action))
                        i += 1
            else:
                # Not enough actions left to form any situation, mark as other
                other_situations.append((current_time, current_action))
                i += 1

    # Handle any remaining action
    if i < len(actions):
        other_situations.append(actions[i])

    return normal_situations, special_situations, other_situations



# if have other situation, label 2; if only have normal situaion, label 0; if have special situaion, label 1
# if label is 0 or 1 return a list contains of timestep in normal and special (sorted) to a csv
# if label 2 retun none
def safe_literal_eval(val):
    try:
        if pd.isna(val):  # Check for NaN
            return {}  # Return an empty dictionary for NaN values
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return {}  # Return an empty dictionary if parsing fails

def analyze_and_output(base_path):
    output_path = os.path.join(base_path, 'uvd_gt.csv')
    results = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file == 'processed_output_zed.csv':
                file_path = os.path.join(root, file)
                data = pd.read_csv(file_path)

                # Process the data
                data['left_pos'] = data['left_pos'].apply(safe_literal_eval)
                data['left_ori'] = data['left_ori'].apply(safe_literal_eval)
                data['right_pos'] = data['right_pos'].apply(safe_literal_eval)
                data['right_ori'] = data['right_ori'].apply(safe_literal_eval)
                data['left_gripper'] = data['left_gripper'].astype(int)
                data['right_gripper'] = data['right_gripper'].astype(int)

                synced_places = data[(data['left_gripper'].shift(1) == 1) & (data['left_gripper'] == 0) &
                                     (data['right_gripper'].shift(1) == 1) & (data['right_gripper'] == 0)]
                if not synced_places.empty:
                    first_sync_time = synced_places.iloc[0]['time']
                    filtered_data = data[data['time'] >= first_sync_time]
                else:
                    filtered_data = data

                left_picks, left_places = track_actions(filtered_data, 'left')
                right_picks, right_places = track_actions(filtered_data, 'right')

                left_paired = pair_actions(left_picks, left_places)
                right_paired = pair_actions(right_picks, right_places)
                left_actions = [(time, 'left_pick') for time, _ in left_paired] + [(time, 'left_place') for _, time in left_paired]
                right_actions = [(time, 'right_pick') for time, _ in right_paired] + [(time, 'right_place') for _, time in right_paired]

                all_actions = left_actions + right_actions
                all_actions.sort()  # Sort by time

                final_actions = merge_actions(all_actions)
                normal, special, other = classify_actions(final_actions)
                # print(final_actions)
                label = 2 if other else 1 if special else 0
                if label in (0, 1):
                    timestamps = [time for sit in normal + special for time in (sit[0], sit[2])]
                    timestamps = sorted(set(timestamps))
                    indices = data[data['time'].isin(timestamps)].index.tolist()
                    folder_name = root.split('/')[-3]  # Assuming the folder structure as described
                    subfolder_name = root.split('/')[-2]
                    subsubfolder_name= root.split('/')[-1]
                    for index, timestamp in zip(indices, timestamps):
                        results.append({'Participant': subfolder_name, 'Garment': subsubfolder_name, 'Index': index, 'Timestamp': timestamp, 'Label': label})
                if label == 2:
                    print(f"Skipping '{file_path}' due to other situations.")
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Output saved to '{output_path}'.")

# Example usage
# base_path = '/media/ubb/T9/useful_data/robot_ssshirt'
# analyze_and_output(base_path)
  







# metrics
# F1 score precison: how many of the predicted are correct (correct uvd_result/total uvd_result), 
# recall: how many of the ground truth are predicted (correct uvd_result/total ground truth)



def read_data(pred_path, gt_path):
    # Read prediction and ground truth data
    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)
    return pred_df, gt_df

def parse_indices(index_str):
    # Convert index string in the format "[1, 2, 3]" to a list of integers
    return ast.literal_eval(index_str)

def compute_metrics_for_line(pred_indices, gt_indices, threshold_lower, threshold_upper):
    # Initialize counters
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    gt_set = set(gt_indices)

    # print("Ground truth set:", gt_set)
    
    if gt_indices:
        min_gt = min(gt_indices)
        pred_indices = [pred for pred in pred_indices if pred > min_gt]
        # print("Filtered predictions:", pred_indices)
    # Create a set for ground truth indices for faster lookup
    for pred in pred_indices:
        # print("Checking prediction:", pred)
        # Initialize a variable to store the matching ground truth index
        matched_gt = None
        # Check if this predicted index matches any ground truth index within the threshold
        for gt in gt_set:
            if gt - threshold_lower <= pred <= gt + threshold_upper:
                true_positives += 1
                matched_gt = gt
                # print(f"True Positive: Prediction {pred} is within the threshold of Ground Truth {gt}")
                break
        if not matched_gt and pred != pred_indices[-1]:
            false_positives += 1
            # print(f"False Positive: Prediction {pred} does not match any ground truth within the threshold.")
    if pred_indices and not matched_gt:
        true_positives += 1  # Assuming the last prediction is always correct
        # print(f"Manually adjusted last prediction {pred_indices[-1]} as True Positive")


    assert true_positives + false_positives == len(pred_indices)
    if true_positives+false_positives ==0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives) 
    recall = true_positives / len(gt_indices)
    # print('len',len(gt_indices))
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall) 
    
    return true_positives,false_positives,precision, recall, f1_score

def analyze_results(pred_df, gt_df, threshold_lower, threshold_upper):
    results = []
    # Iterate over each line in the prediction DataFrame
    for _, pred_row in pred_df.iterrows():
        # Filter ground truth for matching participant and garment
        filtered_gt_df = gt_df[(gt_df['Participant'] == pred_row['Participant']) & (gt_df['Garment'] == pred_row['Garment'])]
        
        if filtered_gt_df.empty:
            # Skip if there is no ground truth data for the participant and garment
            print(f"Skipping {pred_row['Participant']} - {pred_row['Garment']} due to missing ground truth.")
            continue

        gt_indices = filtered_gt_df['Index'].tolist()
        if not gt_indices or len(gt_indices) == 0:
            print(f"Skipping {pred_row['Participant']} - {pred_row['Garment']} due to empty or zero ground truth indices.")
            continue
        if 1 in filtered_gt_df['Label'].values:
            label = 1
        else:
            label = 0
        # Parse the predicted indices
        pred_indices = parse_indices(pred_row['Index'])
        
        # Compute metrics
        true_positives,false_positives,precision, recall, f1_score = compute_metrics_for_line(pred_indices, gt_indices, threshold_lower, threshold_upper)

        # Store results
        results.append({
            'Participant': pred_row['Participant'],
            'Garment': pred_row['Garment'],
            'Camera': pred_row['Camera'],
            'True Positives': true_positives,
            'False Positives': false_positives,
            'Total_pred': true_positives+false_positives,
            'Total_gt': len(gt_indices),
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'Label': label
        })

    return results

# Example usage
gt_path = '/media/ubb/T9/useful_data/robot_selected_item2_resize/uvd_gt.csv'
pred_path = '/media/ubb/T9/useful_data/robot_selected_item2_resize/output.csv'
threshold_lower = 5
threshold_upper = 5

pred_df, gt_df = read_data(pred_path, gt_path)
results = analyze_results(pred_df, gt_df, threshold_lower, threshold_upper)

# Convert results to DataFrame and optionally save or display
results_df = pd.DataFrame(results)
# base_path = '/media/ubb/T9/useful_data/robot_selected_item2_resize'
# results_df.to_csv(os.path.join(base_path,'uvd_phrase2_results.csv'), index=False)








# prases1: find the two hand place timestep, and find the last pick just before the two hand place, we only care before
# select all picks 

def track_actions1(df):
    # Create a boolean series that is True where either left or right gripper changes to 1 (pick) or 0 (place)
    left_changes = df['left_gripper'].diff()
    right_changes = df['right_gripper'].diff()
    
    # Identify picks and places by checking for specific changes in gripper status
    picks = df[(left_changes == 1) | (right_changes == 1)].index.tolist()  # Picks when either changes to 1
    places = df[(left_changes == -1) | (right_changes == -1)].index.tolist()  # Places when either changes to -1
    
    return picks, places

def filter_picks(picks):
    if not picks:
        return []
    filtered_picks = [pick for pick in picks]
    # print(filtered_picks)
    return filtered_picks[1:] if len(filtered_picks) > 1 else []
base_path = '/media/ubb/T9/useful_data/robot_selected_item2_resize'
results1 = []
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file == 'processed_output_zed.csv':
            file_path = os.path.join(root, file)
            data = pd.read_csv(file_path)


# Process the data
            data['left_pos'] = data['left_pos'].apply(safe_literal_eval)
            data['left_ori'] = data['left_ori'].apply(safe_literal_eval)
            data['right_pos'] = data['right_pos'].apply(safe_literal_eval)
            data['right_ori'] = data['right_ori'].apply(safe_literal_eval)
            data['left_gripper'] = data['left_gripper'].astype(int)
            data['right_gripper'] = data['right_gripper'].astype(int)

            synced_places = data[(data['left_gripper'].shift(1) == 1) & (data['left_gripper'] == 0) &
                        (data['right_gripper'].shift(1) == 1) & (data['right_gripper'] == 0)]
            if not synced_places.empty:
                first_sync_time = synced_places.iloc[0]['time']
                filtered_data = data[data['time'] <= first_sync_time]
            else:
                filtered_data = data

            picks, places = track_actions1(filtered_data)

            last_pick_before_sync = max(picks)

            filtered_picks = picks[1:]
            # filtered_picks = filter_picks(picks)

            # print(filtered_picks)

            folder_name = root.split('/')[-2]  # Assuming the folder structure as described
            subfolder_name = root.split('/')[-1]
            # timestamp = data.loc[filtered_picks, 'time']
            results1.append({'Participant': folder_name, 'Garment': subfolder_name, 'Index': filtered_picks})
                
    # Save results to CSV
    df1 = pd.DataFrame(results1)
    # print(df1)
    # # base_path = '/media/ubb/T9/useful_data/robot_selected_item2_resize'
    # output_path = os.path.join(base_path, 'uvd_gt_bf.csv')
    # df1.to_csv(output_path, index=False)









# phrase one: select from first pick to last pick (before two hand place)
# no first pick; pred is less than last pick+10


# metric for phrase one
def read_data(pred_path, gt_path):
    # Read prediction and ground truth data
    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)
    return pred_df, gt_df

def parse_indices(index_str):
    # Convert index string in the format "[1, 2, 3]" to a list of integers
    return ast.literal_eval(index_str)

def compute_metrics_for_line(pred_indices, gt_indices, threshold_lower, threshold_upper):
    # Initialize counters
    true_positives = 0
    false_positives = 0
    # false_negatives = 0
    gt_set = set(gt_indices)
    # print('gt_set',gt_set)
    # print("Ground truth set:", gt_set)
    
    if gt_indices:
        max_gt = max(gt_indices)
        # print('max_gt',max_gt)
        pred_indices = [pred for pred in pred_indices if pred <= max_gt+10]
        # print('pred_indices',pred_indices)
        # print("Filtered predictions:", pred_indices)
    # Create a set for ground truth indices for faster lookup
    for pred in pred_indices:
        # print("Checking prediction:", pred)
        # Initialize a variable to store the matching ground truth index
        matched_gt = None
        # Check if this predicted index matches any ground truth index within the threshold
        for gt in gt_set:
            if gt - threshold_lower <= pred <= gt + threshold_upper:
                true_positives += 1
                matched_gt = gt
                # print(f"True Positive: Prediction {pred} is within the threshold of Ground Truth {gt}")
                break
        if not matched_gt:
            false_positives += 1
            # print(f"False Positive: Prediction {pred} does not match any ground truth within the threshold.")
    # if pred_indices and not matched_gt:
    #     true_positives += 1  # Assuming the last prediction is always correct
        # print(f"Manually adjusted last prediction {pred_indices[-1]} as True Positive")


    assert true_positives + false_positives == len(pred_indices)
    if true_positives + false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives + false_positives) 
    # print('true_positives',true_positives)
    # print('false_positives',false_positives)
    # print('precision',precision)
    recall = true_positives / len(gt_indices)
    # print('recall',recall)
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return true_positives, false_positives, precision, recall, f1_score
    

def analyze_results(pred_df, gt_df, threshold_lower, threshold_upper):
    results = []
    # Iterate over each line in the prediction DataFrame
    for _, pred_row in pred_df.iterrows():
        # Filter ground truth for matching participant and garment
        filtered_gt_df = gt_df[(gt_df['Participant'] == pred_row['Participant']) & (gt_df['Garment'] == pred_row['Garment'])]
        # gt_indices = filtered_gt_df['Index'].tolist()
        gt_indices = parse_indices(filtered_gt_df['Index'].iloc[0]) if not filtered_gt_df.empty else []


        # Parse the predicted indices
        pred_indices = parse_indices(pred_row['Index'])
        
        # Compute metrics
        true_positives,false_positives,precision, recall, f1_score = compute_metrics_for_line(pred_indices, gt_indices, threshold_lower, threshold_upper)

        # Store results
        results.append({
            'Participant': pred_row['Participant'],
            'Garment': pred_row['Garment'],
            'Camera': pred_row['Camera'],
            'True Positives': true_positives,
            'False Positives': false_positives,
            'Total_pred': true_positives+false_positives,
            'Total_gt': len(gt_indices),
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score
        })

    return results

# Example usage
gt_path = '/media/ubb/T9/useful_data/robot_selected_item2_resize/uvd_gt_bf.csv'
pred_path = '/media/ubb/T9/useful_data/robot_selected_item2_resize/output.csv'
threshold_lower = 5
threshold_upper = 5
base_path1 = '/media/ubb/T9/useful_data/robot_selected_item2_resize'
pred_df, gt_df = read_data(pred_path, gt_path)
results_df = analyze_results(pred_df, gt_df, threshold_lower, threshold_upper)

# Convert results to DataFrame and optionally save or display
results_df = pd.DataFrame(results_df)
results_df.to_csv(os.path.join(base_path1,'uvd_phrase1_results.csv'), index=False)

