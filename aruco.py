import numpy as np
import cv2
import cv2.aruco as aruco
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import DMatrix

# Load saved model and scaler
model = joblib.load('path_to_save_model.pkl')
scaler = joblib.load('path_to_save_scaler.pkl')

print("Model and scaler loaded successfully.")

# Prompt the user for input
exercise = input("Choose an exercise (Incline DB Bench or Preacher Curls): ")
height = float(input("Enter your height in inches: "))
weight = float(input("Enter your weight in pounds: "))

# Validate the exercise input and map it to the encoded value
label_mapping = {"Incline DB Bench": 0, "Preacher Curls": 1}
if exercise not in label_mapping:
    raise ValueError("Invalid exercise choice. Please choose either 'Incline DB Bench' or 'Preacher Curls'.")

# Store inputs in a dictionary
user_data = {
    "Exercise": label_mapping[exercise],
    "Rep Time %": None,
    "Height": height,
    "Weight": weight,
}

# Specify the folder containing the video files
video_folder = r"C:\Users\jaidi\Downloads\New folder"  # Replace with the path to your folder

# Initialize the ArUco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()


# Function to detect ArUco markers
def detect_aruco_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # Filter markers to only keep those with ID 48
    if ids is not None:
        ids = ids.flatten()  # Flatten the ids array
        filtered_corners = []
        filtered_ids = []
        for i, marker_id in enumerate(ids):
            if marker_id == 48:  # Filter for marker ID 48
                filtered_corners.append(corners[i])
                filtered_ids.append(marker_id)
        return filtered_corners, np.array(filtered_ids)  # Return as numpy array
    return [], np.array([])  # Return empty arrays if no markers detected


# Process each video file in the folder
for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.MOV', '.mov')):  # Add more video formats if needed
        video_path = os.path.join(video_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        # Variables for boundary lines and rep counting
        observing = True
        observation_start_time = None
        y_positions = []
        upper_line = None
        lower_line = None
        temp_upper_line = None
        temp_lower_line = None
        concentric_times = []
        center_list = []
        concentric_start_time = None
        last_detected_time = 0
        # Initialize rep count
        rep_count = 0

        # State machine variables
        state = "Idle"
        last_detected_frame = None
        current_frame = None

        # Fixed observation time in seconds
        fixed_observation_duration = 2500  # 4 seconds in milliseconds

        # Set desired frame rate (e.g., 24 frames per second)
        desired_fps = 24
        frame_delay = int(100 / desired_fps)  # Delay in milliseconds

        print(f"Processing video file: {video_file}")

        testlist = []
        timelist = []
        while True:
            cv2.namedWindow('ArUco Marker Tracking', cv2.WINDOW_NORMAL)
            ret, frame = cap.read()
            if not ret:
                break

            testlist.append(current_frame)
            timelist.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            # Get the current timestamp in milliseconds
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)

            corners, ids = detect_aruco_markers(frame)

            if len(corners) > 0:
                frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                # Calculate the center of the marker
                centers = [np.mean(corner[0], axis=0) for corner in corners]

                # Store the center of the marker
                current_frame = centers[0][1] if centers else None

                if observing:
                    if observation_start_time is None:
                        observation_start_time = current_timestamp

                    # Collect y positions for the observation period
                    if current_frame is not None:
                        y_positions.append(current_frame)
                        # Keep buffer size to the specified limit
                        if len(y_positions) > 100:  # Arbitrary buffer size
                            y_positions.pop(0)

                    if current_timestamp - observation_start_time >= fixed_observation_duration:
                        observing = False
                        # Calculate the boundary lines
                        y_positions.sort()
                        if len(y_positions) > 1:
                            temp_upper_line = np.mean(y_positions[-2:])
                            temp_lower_line = np.mean(y_positions[:2])
                            margin = (temp_upper_line - temp_lower_line)
                        # Set final boundaries
                        upper_line = temp_lower_line + 0.2 * margin
                        lower_line = temp_upper_line - 0.2 * margin
                        temp_upper_line = None
                        temp_lower_line = None

                else:
                    if upper_line is not None and lower_line is not None:
                        # Draw boundary lines
                        cv2.line(frame, (0, int(upper_line)), (frame.shape[1], int(upper_line)), (0, 255, 0), 2)
                        cv2.line(frame, (0, int(lower_line)), (frame.shape[1], int(lower_line)), (0, 0, 0), 2)

                    if state == "Idle":
                        if last_detected_frame > lower_line and (current_frame is None or current_frame < lower_line):
                            # Concentric phase starts
                            concentric_start_time = last_detected_time
                            state = "Concentric"
                    if state == "Concentric":
                        if last_detected_frame > upper_line and current_frame < upper_line:
                            # Concentric phase ends
                            concentric_end_time = current_timestamp
                            concentric_duration = (
                                                              concentric_end_time - concentric_start_time) / 1000.0  # Convert milliseconds to seconds
                            concentric_times.append(concentric_duration)
                            print(f"Concentric duration: {concentric_duration:.2f} seconds")
                            rep_count += 1
                            print(f"Rep count: {rep_count}")
                            state = "Idle"

                if current_frame is not None:
                    last_detected_frame = current_frame
                    last_detected_time = cap.get(cv2.CAP_PROP_POS_MSEC)

            # Display rep count on the frame
            cv2.putText(frame, f"Reps: {rep_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Check the number of reps and calculate Rep Time %
            if len(concentric_times) < 7:
                cv2.putText(frame, "RIR: More Reps Needed", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)
            else:
                if len(concentric_times) < 5:
                    avg_reps = np.mean(concentric_times[:-1])
                elif len(concentric_times) < 9:
                    avg_reps = np.mean(concentric_times[:4])
                else:
                    last_8_reps = concentric_times[-8:]
                    avg_reps = np.mean(last_8_reps[:4])
                user_data["Rep Time %"] = concentric_times[-1] / avg_reps

                # Convert user_data to DataFrame and prepare for model prediction
                df = pd.DataFrame([user_data])
                df = scaler.transform(df)

                label_mapping_reverse = {0: "4+", 1: "2-3", 2: "0-1"}

                # Convert to DMatrix for XGBoost
                dmatrix = DMatrix(df)

                # Predict RIR using the model
                predictions = model.predict(dmatrix)

                if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
                    # Get the index of the max probability
                    predicted_class_index = np.argmax(predictions, axis=1)[0]

                    # Map the predicted class index to the RIR category
                    rir_category = label_mapping_reverse.get(predicted_class_index, "Unknown")

                    # print(f"Predicted RIR Category: {rir_category}")
                else:
                    print("Unexpected prediction output:", predictions)
                    rir_category = None  # or handle as appropriate

                # Display the RIR on the video frame
                cv2.putText(frame, f"RIR: {rir_category}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                            cv2.LINE_AA)

            cv2.imshow('ArUco Marker Tracking', frame)
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            if key == ord('e') and state == "Concentric":
                # Manually end the concentric rep
                concentric_end_time = current_timestamp
                concentric_duration = (
                                                  concentric_end_time - concentric_start_time) / 1000.0  # Convert milliseconds to seconds
                concentric_times.append(concentric_duration)
                print(f"Concentric duration: {concentric_duration:.2f} seconds")
                rep_count += 1
                print(f"Rep count: {rep_count}")
                state = "Idle"

        cap.release()
        cv2.destroyAllWindows()