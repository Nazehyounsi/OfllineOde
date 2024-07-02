
import torch
import os
import numpy as np
from sklearn.neighbors import KernelDensity
import pandas as pd
from Model import Model_mlp_diff, Model_Cond_Diffusion, ObservationEmbedder, SpeakingTurnDescriptorEmbedder, ChunkDescriptorEmbedder
import time
import random


def generate_random_series_sequence(sequence_length, max_series=4):
    # Initialize the sequence with zeros
    sequence = [0] * sequence_length

    # Define the values that series can take
    possible_values = [1, 2, 3]

    # Calculate the maximum length for a series (half of the global sequence length)
    max_series_length = sequence_length // 2

    # Initialize a counter for the number of series added
    series_count = 0

    while series_count < max_series:
        # Choose a random value for the series
        value = random.choice(possible_values)

        # Choose a random length for the series, at least 5 frames and at most max_series_length
        series_length = random.randint(5, max_series_length)

        # Choose a random start position for the series
        start_position = random.randint(0, sequence_length - series_length)

        # Fill the sequence with the value for the length of the series
        for i in range(start_position, start_position + series_length):
            sequence[i] = value

        # Increment the counter for the number of series added
        series_count += 1

    return sequence


# Example usage:
sequence_length = 100  # Length of the sequence
randomized_sequence = generate_random_series_sequence(sequence_length)
print("Randomized Sequence:", randomized_sequence)
n_hidden = 512
n_T = 500
num_event_types =13
event_embedding_dim =64
embed_output_dim =128
guide_w =0
num_facial_types = 7
facial_embed_dim = 32
cnn_output_dim = 512  # Output dimension after passing through CNN layers
lstm_hidden_dim = 256
sequence_length = 137
average_duration = 137 * 0.04 #25 frame/Sec

def is_valid_chunk(chunk):
    if not chunk[0]:  # Check if the chunk[0] is an empty list
        return False
    for event in chunk[0]:
        if event[0] is float or event[1] is None or event[2] is None:
            return False
    return True
def transform_action_to_sequence(events, sequence_length):
    # This remains the same as the transform_to_sequence function
    sequence = [0] * sequence_length
    for event in events:

        event_type, start_time, duration = event
        start_sample = int(start_time * sequence_length)
        end_sample = int(start_sample + (duration * sequence_length))
        for i in range(start_sample, min(end_sample, sequence_length)):
            sequence[i] = event_type
    return sequence

def temporal_to_frame_sequence(temporal_predictions, speaking_turn_duration, fps=25):
    # Initialize the sequence with zeros
    sequence_length = int(speaking_turn_duration * fps)
    real_sequence = [0] * sequence_length

    for category, start_time, duration in temporal_predictions:
        # Convert start_time and duration from seconds to frame indices
        start_frame = int(round(start_time * fps))
        end_frame = int(round((start_time + duration) * fps))

        # Fill in the frames for this category
        for frame in range(start_frame, min(end_frame, sequence_length)):
            real_sequence[frame] = category

    return real_sequence


def transform_obs_to_sequence(events, sequence_length):
    facial_expression_events = [21, 27, 31]  # Define facial expression event types
    sequence = [0] * sequence_length
    mi_behaviors = []  # To store MI behaviors
    for event in events:
        event_type, start_time, duration = event
        if event_type not in facial_expression_events and event_type == round(event_type):
            mi_behaviors.append(event_type)
        else:
            start_sample = int(start_time * sequence_length)
            end_sample = int(start_sample + (duration * sequence_length))
            for i in range(start_sample, min(end_sample, sequence_length)):
                if event_type == round(event_type):
                    sequence[i] = event_type
                else:
                    sequence[i] = 0
    return sequence, mi_behaviors


#SCALING THROUGH TUPLE CONVERsion
def transcribe_sequence_to_temporal_format(sequence, fps=25):

    sequence = list(sequence)  # Convert sequence to list if it's a NumPy array
    sequence.append("END")  # Add a terminator to flush the last category

    temporal_sequence = []
    current_category = sequence[0]
    start_frame = 0

    # if all(category == current_category for category in sequence):
    #     # If the sequence consists of only one category, handle it directly
    #     temporal_sequence.append((current_category, 0.0, speaking_turn_duration))
    # else:
    for i, category in enumerate(sequence, 1):
        if category != current_category or category == "END":
            duration = (i - start_frame - 1) / fps  # Adjust for the off-by-one from enumeration start at 1
            start_time = start_frame / fps
            if category != "END":  # Avoid adding the "END" marker to the temporal sequence
                temporal_sequence.append((current_category, start_time, duration))
            current_category = category
            start_frame = i

    return temporal_sequence


def scale_temporal_sequence(temporal_sequence, scaling_ratio, speaking_turn_duration):
    """
    Scale the start times and durations of a temporal sequence by a given scaling ratio.
    """
    if not temporal_sequence:
        scaled_sequence = [(0, 0.0, speaking_turn_duration)]

    else :
        scaled_sequence = [(category, start_time * scaling_ratio, duration * scaling_ratio) for category, start_time, duration in temporal_sequence]
    return scaled_sequence


def append_sequences_to_file(observation_sequence, action_sequence, speaking_turn_duration, file_path):

    with open(file_path, 'a') as file:  # Open file in append mode
        # Convert sequences to string format
        obs_seq_str = ' '.join(str(item) for item in observation_sequence)
        act_seq_str = ' '.join(str(item) for item in action_sequence)
        # Convert speaking turn duration to string format
        speaking_turn_duration_str = f"Duration: {speaking_turn_duration:.2f} seconds"

        # Append the observation sequence, then the action sequence, then an empty line
        file.write(speaking_turn_duration_str + '\n')
        file.write(obs_seq_str + '\n')
        file.write(act_seq_str + '\n\n')  # Two newlines to add an empty line between chunks

def append_sequence_to_csv(sequence, output_csv_path, frame_rate=25):
    # Calculate the duration of each frame
    duration_per_frame = 1.0 / frame_rate

    # Determine the starting timestamp for this sequence
    if os.path.exists(output_csv_path):
        # Load the existing data
        existing_df = pd.read_csv(output_csv_path)
        # Assume the last timestamp is the one at the end of the file
        last_timestamp = existing_df['timestamp'].iloc[-1]
    else:
        # If file doesn't exist, start from timestamp zero
        last_timestamp = 0.0
        # Also create a DataFrame with appropriate column names
        existing_df = pd.DataFrame(columns=['timestamp'] + [str(i) for i in range(4)])
        existing_df.to_csv(output_csv_path, index=False)

    # Prepare new data to be appended
    new_data = []
    for category in sequence:
        # Create a new row for each item in the sequence
        new_row = [last_timestamp + duration_per_frame] + [1 if i == category else 0 for i in range(4)]
        new_data.append(new_row)
        # Increment the timestamp for the next row
        last_timestamp += duration_per_frame

    # Convert new data into a DataFrame
    new_df = pd.DataFrame(new_data, columns=['timestamp'] + [str(i) for i in range(4)])

    # Append new data to the existing CSV file without reading it into DataFrame
    new_df.to_csv(output_csv_path, mode='a', header=False, index=False)

    # Return the new data as DataFrame (optional)
    return new_df


def process_single_file(file_path):
    facial_expression_mapping = {0: 0, 16: 1, 26: 2, 30: 3, 21: 4, 27: 5, 31: 6}
    mi_behavior_mapping = {39: 1, 38: 2, 40: 3, 41: 4, 3: 5, 4: 6, 5: 7, 6: 8, 8: 9, 11: 10, 13: 11, 12: 12}

    # Step 1: Load and preprocess the sample
    raw_data = load_data_from_file(file_path)  # Assumes this function is correctly adapted for single files
    processed_data, sequence_length = preprocess_data(raw_data)  # Wrap in list, assuming preprocessing expects a list

    # Initialize a container for results
    evaluation_results = []
    all_chunks = []
    max_z_len = 3

    # Step 2: Transform data to sequences, including the chunk_descriptor
    for (observation, action, chunk_descriptor), speaking_turn_duration in processed_data:
        x, z = transform_obs_to_sequence(observation, sequence_length)
        y = transform_action_to_sequence(action, sequence_length)

        if len(z) < max_z_len:
            z = z + [0] * (max_z_len - len(z))  # Assuming 0 is an appropriate padding value

        # Reassign event values based on the new mappings
        x = [facial_expression_mapping.get(item, item) for item in x]
        y = [facial_expression_mapping.get(item, item) for item in y]
        z = [mi_behavior_mapping.get(item, item) for item in z]  # Assuming 'z' needs mapping as well

        # Convert to tensors or your model's required input format, ensuring a batch dimension
        x_tensor = torch.tensor([x], dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        z_tensor = torch.tensor([z], dtype=torch.float32)
        chunk_descriptor_tensor = torch.tensor([chunk_descriptor], dtype=torch.float32)
        speaking_turn_duration_tensor = torch.tensor([speaking_turn_duration], dtype=torch.float32)

        # Append a tuple for each chunk to the list
        all_chunks.append((x_tensor, y_tensor, z_tensor, chunk_descriptor_tensor, speaking_turn_duration_tensor))
    return all_chunks


def load_data_from_file(file_path):
    if not os.path.exists(file_path) or not file_path.endswith(".txt"):
        raise ValueError("File does not exist or is not a '.txt' file.")

    with open(file_path, 'r') as f:
        lines = f.readlines()
        non_empty_lines = [line.strip() for line in lines if line.strip() != ""]
        chunks = [eval(line) for line in non_empty_lines]


    observation_chunks = [chunk[:-1] for chunk in chunks[::2]]  # get all tuples except the last one
    action_chunks = chunks[1::2]  # extract every second element starting from 1
    chunk_descriptors = [chunk[-1] for chunk in chunks[::2]]


    for i in range(len(chunk_descriptors)):
        event = list(chunk_descriptors[i])
        if event[2] is None:
            event[2] = -1
        if event[1] is None:
            event[1] = -1
        chunk_descriptors[i] = tuple(event)

    all_data = list(zip(observation_chunks, action_chunks, chunk_descriptors))

    return all_data

def preprocess_data(data):
    filtered_data = [chunk for chunk in data if is_valid_chunk(chunk)]
    sequence_length = 137  # Assuming 25 FPS

    processed_chunk = []

    for i, chunk in enumerate(filtered_data):
        if not chunk[0]:  # Skip if the observation vector is empty
            continue

        # Combine and filter start times and durations for observation and action events
        combined_events = chunk[0] + chunk[1]
        valid_events = [(event[1], event[2]) for event in combined_events if
                        isinstance(event[1], float) and event[1] > 0]

        if not valid_events:  # Skip if no valid start times
            continue

        # Calculate the minimum start time and the maximum end time
        min_start_time = min(event[0] for event in valid_events)
        max_end_time = max((event[0] + event[1] for event in valid_events), default=min_start_time)
        speaking_turn_duration = max_end_time - min_start_time

        # Adjust speaking turn duration based on next chunk's starting time
        if i + 1 < len(filtered_data):
            next_chunk = filtered_data[i + 1]
            next_chunk_events = next_chunk[0] + next_chunk[1]
            next_chunk_start_times = [event[1] for event in next_chunk_events if
                                      isinstance(event[1], float) and event[1] > 0]

            if next_chunk_start_times:
                next_chunk_start_time = min(next_chunk_start_times)
                current_interaction_time = min_start_time + speaking_turn_duration
                if current_interaction_time > next_chunk_start_time:
                    speaking_turn_duration = next_chunk_start_time - min_start_time


        if speaking_turn_duration <= 0: # Skip turns with non-positive duration
            continue

        # Normalize start times and durations within each chunk
        for vector in [0, 1]:  # 0 for observation, 1 for action
            for i, event in enumerate(chunk[vector]):
                event_type, start_time, duration = event
                if start_time == 0.0:
                    continue
                if start_time<min_start_time:
                    start_time = min_start_time
                # Standardize the starting times relative to the speaking turn's start
                normalized_start_time = (start_time - min_start_time)


                # Normalize start times and durations against the speaking turn duration
                normalized_start_time = normalized_start_time / speaking_turn_duration
                normalized_duration = duration / speaking_turn_duration


                # Update the event with normalized values
                chunk[vector][i] = (event_type, round(normalized_start_time, 3), round(normalized_duration, 3))

        processed_chunk.append((chunk, speaking_turn_duration))
    return processed_chunk, sequence_length


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Process a single file and prepare data for the model
    #file_path = 'C:/Users/NEZIH YOUNSI/Desktop/Hcapriori_input/Observaton_Context_Tuples/interview_36_hcapriori__processed_to_OCTuple.txt'
    #file_path = 'interview_36_hcapriori__processed_to_OCTuple.txt'
    # Assuming 'process_single_file' returns data ready for model input
    file_path = 'intervieww_36_606.txt'

    all_chunks = process_single_file(file_path)

    # Assuming all_chunks is populated as described earlier
    x_dims = [chunk[0].shape[-1] for chunk in all_chunks]  # Assuming the last dimension is the feature size
    y_dims = [chunk[1].shape[-1] for chunk in all_chunks]
    # Find max dimensions if variable; otherwise, this confirms the static dimensions
    x_dim = max(x_dims)
    y_dim = max(y_dims)



    observation_embedder = ObservationEmbedder(num_facial_types, facial_embed_dim, cnn_output_dim, lstm_hidden_dim,
                                               sequence_length)
    mi_embedder = SpeakingTurnDescriptorEmbedder(num_event_types, event_embedding_dim, embed_output_dim)
    chunk_embedder = ChunkDescriptorEmbedder(continious_embedding_dim=16, valence_embedding_dim=8, output_dim=64)
    model_path = 'saved_model_NewmodelChunkd.pth'
    #model_path = 'saved_model_NewmodelChunkdCFG.pth'
    nn_model = Model_mlp_diff(
        observation_embedder, mi_embedder, chunk_embedder, sequence_length, net_type="transformer")
    model = Model_Cond_Diffusion(
        nn_model,
        observation_embedder,
        mi_embedder,
        chunk_embedder,
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        x_dim=x_dim,
        y_dim=y_dim,
        drop_prob=0,
        guide_w=guide_w,
    )

    model.load_state_dict(torch.load(model_path, map_location=device))# Load the trained weights

    #model.load_state_dict(torch.load(model_path))  # Load the trained weights
    model.to(device)

    model.eval()  # Set the model to evaluation mode

    guide_weight = guide_w
    kde_samples = 3 # Example: Number of samples to generate for KDE

    real_actions =[]
    # Iterate over each chunk in the file
    for x_tensor, y_tensor, z_tensor, chunk_descriptor_tensor, speaking_turn_duration_tensor in all_chunks:
        # Convert tensors to the device
        x_batch = x_tensor.to(device)
        z_batch = z_tensor.to(device)
        y_batch = y_tensor.to(device)
        chunk_descriptor = chunk_descriptor_tensor.to(device)
        speaking_turn_duration = speaking_turn_duration_tensor.item()

        # print("la sequence obs :")
        # print(x_batch[0])
        # print("la target :")
        # print(y_batch[0])
        # print("les Mi behaviors :")
        # print(z_batch[0])
        # print("la durée du chunk : ")
        # print(speaking_turn_duration)

        # Generate multiple predictions for KDE
        all_predictions = []
        for _ in range(kde_samples):
            with torch.no_grad():
                model.guide_w = guide_weight
                start_time = time.time()
                y_pred = model.sample(x_batch, z_batch, chunk_descriptor).detach().cpu().numpy()
                end_time = time.time()
                inference_time = end_time - start_time
                print(f"Inference time for the current batch: {inference_time: .4f} seconds")
                all_predictions.append(y_pred)

        # Apply KDE for the chunk and determine the best prediction
        best_prediction = np.zeros_like(y_pred)
        single_pred_samples = np.array(all_predictions).squeeze()
        kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(single_pred_samples)
        log_density = kde.score_samples(single_pred_samples)
        best_idx = np.argmax(log_density)
        best_prediction = single_pred_samples[best_idx]

        # Post-process thae best prediction
        best_prediction = np.round(best_prediction)
        best_prediction[best_prediction == 4] = 3
        best_prediction[best_prediction < 0] = 0

        print("la sequence obs :")
        print(x_batch[0])
        # print("la target :")
        # print(y_batch[0])
        print("la prediction :")
        print(best_prediction)
        print("la durée du chunk :")
        print(speaking_turn_duration)

        scaling_ratio = speaking_turn_duration / average_duration
        obs_seq = x_batch[0].cpu().numpy()
        #target_seq = y_batch[0].cpu().numpy()
        #best_prediction=target_seq

        temporal_obs_sequence = transcribe_sequence_to_temporal_format(obs_seq)
        scaled_obs_sequence = scale_temporal_sequence(temporal_obs_sequence, scaling_ratio, speaking_turn_duration)
        print("observation:", scaled_obs_sequence)

        temporal_act_sequence = transcribe_sequence_to_temporal_format(best_prediction)
        scaled_act_sequence = scale_temporal_sequence(temporal_act_sequence, scaling_ratio, speaking_turn_duration)
        print("Best Prediction:", scaled_act_sequence)
        save_file = '36_txt_compar.txt'
        append_sequences_to_file(scaled_obs_sequence, scaled_act_sequence, speaking_turn_duration, save_file)
        real_action_sequence = temporal_to_frame_sequence(scaled_act_sequence, speaking_turn_duration, fps = 25)
        real_action_sequence = generate_random_series_sequence(len(real_action_sequence))
        print("real prediction:", real_action_sequence)
        output_csv_path = 'intermed.csv'
        df = append_sequence_to_csv(real_action_sequence, output_csv_path)

        #Headers renaming
        # Load your CSV file into a DataFrame
        df = pd.read_csv(output_csv_path)
        # Rename columns according to the provided dictionary
        rename_dict = {'1': 'AU12_r', '2': 'AU09_r', '3': 'AU15_r'}
        df.rename(columns=rename_dict, inplace=True)
        # Drop the column labeled as '0'
        df.drop(columns='0', inplace=True)
        # Save the transformed DataFrame back to CSV
        df.to_csv('36_csv_file.csv', index=False)


if __name__ == '__main__':
    main()