#!/usr/bin/env python3
from controller import Supervisor, Motor
import cv2
import mediapipe as mp
import math
import time
from collections import deque  # For moving average

# Initialize Webots
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# YouBot Motors (four mecanum wheels)
wheel_names = ["wheel1", "wheel2", "wheel3", "wheel4"]  # front-left, front-right, back-left, back-right
wheels = [robot.getDevice(name) for name in wheel_names]
for wheel in wheels:
    wheel.setPosition(float('inf'))
    wheel.setVelocity(0.0)
max_speed = 10.0  # YouBot max speed in rad/s

# YouBot Arm and Gripper
arm_names = ["arm1", "arm2", "arm3", "arm4", "arm5"]
arms = [robot.getDevice(name) for name in arm_names]
gripper_names = ["finger::left", "finger::right"]
grippers = [robot.getDevice(name) for name in gripper_names]
for arm in arms:
    arm.setVelocity(0.5)
for gripper in grippers:
    gripper.setVelocity(0.5)

# Arm positions for pick and place
PICK_POSITION = [3.14159, 1.2, -1.0, 1.3, 0.0]  # Adjusted to stay within joint limits and avoid WoodenBox
PLACE_POSITION = [3.14159, 0.5, -0.5, 0.0, 0.0]  # Adjusted to place on carrier
GRIPPER_OPEN = 0.025
GRIPPER_CLOSE = 0.0

# Rotate the YouBot 180 degrees at the start
youbot_node = robot.getFromDef("youbot")
if youbot_node is None:
    print("Error: Could not find node with DEF 'youbot'")
else:
    rotation_field = youbot_node.getField("rotation")
    rotation_field.setSFRotation([0, 0, 1, 3.14159])  # Rotate 180 degrees
    print("Rotated YouBot 180 degrees")

# MediaPipe Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# State variables
last_gesture = None
gesture_cooldown = 0
holding_box = False
moving_to_pick = False
moving_to_place = False
target_x = 1.0  # Initial YouBot x position (matches world file)
target_y = 0.0  # Initial YouBot y position
movement_start_time = 0  # For timeout
last_position = [1.0, 0.0]  # For collision detection (matches world file)
position_change_threshold = 0.001  # Increased to 1 mm
movement_timeout = 10.0  # Timeout in seconds
min_steps_before_collision_check = 100  # Wait longer before checking for collision
step_counter = 0  # To count steps since movement started
small_movement_counter = 0  # To count consecutive steps with small position changes
small_movement_threshold = 50  # Require 50 steps (~1.6s) for collision detection
position_changes = deque(maxlen=10)  # Moving average of last 10 position changes

# Main Loop
while robot.step(timestep) != -1:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    gesture = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark
            wrist = landmarks[mp_hands.HandLandmark.WRIST]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]
            thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
            index_mcp = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            middle_mcp = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            ring_mcp = landmarks[mp_hands.HandLandmark.RING_FINGER_MCP]
            pinky_mcp = landmarks[mp_hands.HandLandmark.PINKY_MCP]

            def dist(lm1, lm2):
                return math.sqrt((lm1.x - lm2.x)**2 + (lm1.y - lm2.y)**2 + (lm1.z - lm2.z)**2)
            reference_length = dist(wrist, middle_mcp)
            thumb_dist = dist(thumb_tip, thumb_mcp)
            index_dist = dist(index_tip, index_mcp)
            middle_dist = dist(middle_tip, middle_mcp)
            ring_dist = dist(ring_tip, ring_mcp)
            pinky_dist = dist(pinky_tip, pinky_mcp)
            extended_threshold = 0.6 * reference_length
            curled_threshold = 0.55 * reference_length
            
            # Thumbs up: Thumb extended, others curled
            if (thumb_dist > extended_threshold and index_dist < curled_threshold and
                middle_dist < curled_threshold and ring_dist < curled_threshold and pinky_dist < curled_threshold):
                gesture = "thumbs_up"
            # Fist: All fingers curled
            elif (thumb_dist < curled_threshold and index_dist < curled_threshold and
                  middle_dist < curled_threshold and ring_dist < curled_threshold and pinky_dist < curled_threshold):
                gesture = "fist"

    if gesture_cooldown > 0:
        gesture_cooldown -= 1
    elif gesture and gesture != last_gesture:
        print(f"Gesture: {gesture}")
        if gesture == "thumbs_up" and not holding_box:
            # Move to pick position (0.8, 0, 0.09) - Adjusted to avoid arm collision with WoodenBox
            target_x = 0.7
            target_y = 0.0
            moving_to_pick = True
            moving_to_place = False
            movement_start_time = robot.getTime()
            step_counter = 0
            small_movement_counter = 0
            position_changes.clear()
            print("Moving to pick position: (0.8, 0, 0.09)")
        elif gesture == "fist" and holding_box:
            # Move sideways to place position (0.8, 0.5, 0.09)
            target_x = 0.7
            target_y = 0.5
            moving_to_pick = False
            moving_to_place = True
            movement_start_time = robot.getTime()
            step_counter = 0
            small_movement_counter = 0
            position_changes.clear()
            print("Moving to place position: (0.8, 0.5, 0.09)")

        last_gesture = gesture
        gesture_cooldown = int(500 / timestep)  # ~0.5s cooldown

    # Movement logic
    youbot_node = robot.getFromDef("youbot")
    if youbot_node is None:
        print("Error: Could not find node with DEF 'youbot'")
        continue
    current_x = youbot_node.getField("translation").getSFVec3f()[0]
    current_y = youbot_node.getField("translation").getSFVec3f()[1]

    # Check if the YouBot has reached the target position
    if moving_to_pick or moving_to_place:
        x_diff = target_x - current_x
        y_diff = target_y - current_y
        distance = math.sqrt(x_diff**2 + y_diff**2)
        position_change = math.sqrt((current_x - last_position[0])**2 + (current_y - last_position[1])**2)
        position_changes.append(position_change)
        avg_position_change = sum(position_changes) / len(position_changes) if position_changes else position_change
        print(f"Current position: ({current_x}, {current_y}), Target: ({target_x}, {target_y}), Distance: {distance}, Avg position change: {avg_position_change}")

        # Check for timeout
        current_time = robot.getTime()
        if (current_time - movement_start_time) > movement_timeout:
            print("Movement timeout reached, stopping YouBot")
            for wheel in wheels:
                wheel.setVelocity(0.0)
            if moving_to_pick:
                moving_to_pick = False
            elif moving_to_place:
                moving_to_place = False
            continue

        # Check for collision using average position change
        step_counter += 1
        if step_counter > min_steps_before_collision_check:
            # Check if the YouBot is trying to move (non-zero velocity)
            wheel_velocity = abs(wheels[0].getVelocity())  # Check one wheel as a proxy
            if wheel_velocity > 0.1:  # Only check for collision if moving
                if avg_position_change < position_change_threshold:
                    small_movement_counter += 1
                else:
                    small_movement_counter = 0
                if small_movement_counter >= small_movement_threshold:
                    print("Detected possible collision (consistent small average movement), stopping YouBot")
                    for wheel in wheels:
                        wheel.setVelocity(0.0)
                    if moving_to_pick:
                        moving_to_pick = False
                    elif moving_to_place:
                        moving_to_place = False
                    continue

        if distance > 0.05:  # Relaxed stopping condition
            # Normalize direction
            if distance != 0:
                x_speed = (x_diff / distance) * max_speed
                y_speed = (y_diff / distance) * max_speed
            else:
                x_speed = 0
                y_speed = 0

            # Mecanum wheel velocities for forward/backward (x) and sideways (y) movement
            wheels[0].setVelocity(-x_speed + y_speed)  # front-left
            wheels[1].setVelocity(-x_speed - y_speed)  # front-right
            wheels[2].setVelocity(-x_speed - y_speed)  # back-left
            wheels[3].setVelocity(-x_speed + y_speed)  # back-right
        else:
            # Stop the YouBot
            for wheel in wheels:
                wheel.setVelocity(0.0)
            print("Reached target position")

            if moving_to_pick:
                # Pick the KukaBox
                for gripper in grippers:
                    gripper.setPosition(GRIPPER_OPEN)
                for i, arm in enumerate(arms):
                    arm.setPosition(PICK_POSITION[i])
                print("Opening gripper and lowering arm to pick")
                time.sleep(2)  # Wait for arm to lower
                for gripper in grippers:
                    gripper.setPosition(GRIPPER_CLOSE)
                print("Closing gripper to grasp")
                holding_box = True
                moving_to_pick = False
            elif moving_to_place:
                # Place the KukaBox on the carrier
                for i, arm in enumerate(arms):
                    arm.setPosition(PLACE_POSITION[i])
                print("Moving arm to place position (on carrier)")
                time.sleep(2)  # Wait for arm to move
                for gripper in grippers:
                    gripper.setPosition(GRIPPER_OPEN)
                print("Opening gripper to release on carrier")
                holding_box = False
                moving_to_place = False
    else:
        # Ensure wheels are stopped when not moving
        for wheel in wheels:
            wheel.setVelocity(0.0)

    # Update last position for collision detection
    last_position = [current_x, current_y]

    cv2.imshow("Webcam Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
