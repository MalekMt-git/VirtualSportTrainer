import cv2
import mediapipe as mp
import time
import pyautogui
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)
# Initialize mediapipe drawing class, to draw the landmarks.
mp_drawing = mp.solutions.drawing_utils
# Capture video from your webcam.
video = True
screenview = False

if video:
    cap = cv2.VideoCapture('trimmed.mp4')
else:
    cap = cv2.VideoCapture(1)

# Time control variables
start_time = None
run_time_sec = 5  # Change this value to record for a different duration

# File paths
teacher_sport_data_file = 'teacher_sport_data.txt'
student_sport_data_file = 'student_sport_data.txt'

# Flags to control recording
record_teacher = False
record_student = not record_teacher  # Change this to True when recording the student

if record_teacher:
    open(teacher_sport_data_file, 'w').close()
open(student_sport_data_file, 'w').close()

landmarks_index = [i for i in range(0, 33)]
def fill_data(data_str, file_path):
    # Write the string of landmark data to the file
    with open(file_path, 'a') as file:
        file.write(data_str + '\n')
def normalize_landmarks(landmarks):
    # Calculate the midpoint between the hips as the reference
    reference_point = (landmarks[23] + landmarks[24]) / 2
    normalized_landmarks = landmarks - reference_point
    return normalized_landmarks

def calculate_joint_angles(landmarks):
    angles = []
    for i in range(1, len(landmarks) - 1):
        # Calculate angle for every consecutive triplet of points
        ba = landmarks[i - 1] - landmarks[i]
        bc = landmarks[i + 1] - landmarks[i]
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip values for numerical stability
        angles.append(np.degrees(angle))
    return angles

def calculate_movement_magnitude(landmarks):
    # Calculate the total movement magnitude to differentiate static and dynamic poses
    diffs = np.diff(landmarks, axis=0)
    mag = np.linalg.norm(diffs, axis=2)
    return np.sum(mag)

def calculate_accuracy_of_student_sport(data_teacher, data_student):
    # Convert string data back to numpy arrays and reshape
    data_teacher = np.array([np.fromstring(frame, sep=' ') for frame in data_teacher.split('\n') if frame.strip() != '']).reshape(-1, 33, 3)
    data_student = np.array([np.fromstring(frame, sep=' ') for frame in data_student.split('\n') if frame.strip() != '']).reshape(-1, 33, 3)

    # Normalize landmarks
    data_teacher = np.array([normalize_landmarks(frame) for frame in data_teacher])
    data_student = np.array([normalize_landmarks(frame) for frame in data_student])

    # Convert positions to joint angles
    teacher_angles = np.array([calculate_joint_angles(frame) for frame in data_teacher])
    student_angles = np.array([calculate_joint_angles(frame) for frame in data_student])

    # Calculate the movement magnitude for both teacher and student
    teacher_movement_magnitude = calculate_movement_magnitude(data_teacher)
    student_movement_magnitude = calculate_movement_magnitude(data_student)

    # Calculate the DTW distance between the two sequences of angles
    distance, path = fastdtw(teacher_angles, student_angles, dist=euclidean)

    # Normalize the distance to get a similarity score between 0 and 1
    max_dist = 360  # Maximum possible difference per angle
    max_possible_dist = max_dist * max(len(teacher_angles), len(student_angles))
    similarity = max(0, 1 - (distance / max_possible_dist))

    # Convert the similarity to a percentage
    accuracy = similarity * 100

    # Adjust accuracy based on movement magnitude (This threshold can be adjusted or made dynamic)
    movement_threshold = 10
    if teacher_movement_magnitude < movement_threshold or student_movement_magnitude < movement_threshold:
        accuracy *= (teacher_movement_magnitude + student_movement_magnitude) / (2 * movement_threshold)

    return accuracy

while cap.isOpened() or screenview or video:
    if start_time is None:
        start_time = time.time()  # Start the timer

    elapsed_time = time.time() - start_time
    print(elapsed_time)
    if elapsed_time > run_time_sec:
        break  # Stop after running for run_time_sec

    if not screenview:
        ret, frame = cap.read()
        if not ret:
            if video:
                break
            else:
                continue

        # Convert the frame from BGR to RGB (since mediapipe needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # Process the image and find the pose.
        results = pose.process(frame_rgb)
        # frame = frame_rgb
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    if screenview:
        screenshot = pyautogui.screenshot()
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Convert the frame from BGR to RGB (since mediapipe needs RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  #
        # Process the image and find the pose.
        results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw the pose annotation on the image.
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, landmark in enumerate(results.pose_landmarks.landmark):
            # Calculate the x and y coordinates.
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            # Display the landmark number.
            cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Record data
        if record_teacher or record_student:
            selected_landmarks = [results.pose_landmarks.landmark[i] for i in landmarks_index]  # Shoulders and elbows
            data_str = ' '.join(f"{lm.x} {lm.y} {lm.z}" for lm in selected_landmarks)
            if record_teacher:
                fill_data(data_str, teacher_sport_data_file)
            if record_student:
                fill_data(data_str, student_sport_data_file)
    # Display the resulting frame.
    desired_width = 1600  # Change this to your preferred width
    desired_height = 900  # Change this to your preferred height
    resized_frame = cv2.resize(frame, (desired_width, desired_height))
    cv2.imshow('Screen Pose', resized_frame)
    # Break the loop when 'q' is pressed.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Calculate and print the accuracy after the loop is over
if record_student:
    with open(teacher_sport_data_file, 'r') as teacher_file, open(student_sport_data_file, 'r') as student_file:
        teacher_data = teacher_file.read()  # Read teacher data
        student_data = student_file.read()  # Read student data
        accuracy = calculate_accuracy_of_student_sport(teacher_data, student_data)
        print(f"Accuracy of student's sport: {accuracy}%")
# Release resources
cap.release()
cv2.destroyAllWindows()
