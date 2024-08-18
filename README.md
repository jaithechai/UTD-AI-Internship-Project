# UTD-AI-Internship-Project
Final project from the UTD AI Workshop: Gym Buddy RIR Calculator

Presentation Link: https://docs.google.com/presentation/d/1owSh0jI6tgAfvqnGZK7HgLIHJDzugIt-H90B_Xrgmfc/edit?usp=sharing

## Introduction

Training straight to failure is often considered a poor strategy for beginners. Staying within a certain range of Reps In Reserve (RIR) can be more effective and safer. This project uses AI to help beginners estimate their optimal RIR, balancing intensity and safety.

## Problem + Solution

### Problem Statement

How can beginners accurately gauge their reps in reserve (RIR) during workouts?

### Solution

Using concentric rep times extracted from a video submitted by a user, this project predicts how many reps were left until failure. The AI model processes video data to provide an estimation of RIR, helping users optimize their workout intensity.

Since the necessary data was not readily available online, a team and I collected it ourselves using ArUco stickers for precise tracking and measurement.

## Data/Feature Collection

To train and validate the model, the following features are collected:

- **Rep Time:** Time taken to complete each rep, measured using ArUco stickers.
- **Exercise Type:** Type of exercise performed (e.g., ‘Incline Dumbbell Bench’, ‘Preacher Curls’).
- **RIR Classification:**
  - 0-1 RIR
  - 2-3 RIR
  - 4+ RIR
- **Height in Inches:** User-entered height (e.g., 68 inches).
- **Weight in Pounds:** User-entered weight (e.g., 175 pounds).

## Project Overview

- **Internship Duration:** June 3 - August 2, 2024 (with a break from July 1-5)
- **Instructors:** Dr. Anurag Nagar, Dr. Anjum Chida, and UTD PhD students
- **Final Project Achievement:** Placed in the top 10 out of 40 groups

## Features

- AI-based estimation of Reps In Reserve (RIR) for beginners.
- Utilizes machine learning models to predict optimal training intensity based on video analysis.
- Data collected using ArUco stickers for accurate measurement of concentric rep times.
- User-friendly interface for tracking and adjusting workouts.

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/yourusername/UTD-AI-Internship-Project.git
    ```
2. Navigate into the project directory:
    ```bash
    cd UTD-AI-Internship-Project
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Run the main script to start the application:
    ```bash
    python main.py
    ```
2. Follow the instructions in the command line interface or application window to input workout data and receive RIR estimations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **UT Dallas** for providing the opportunity and resources for the AI Internship.
- **Dr. Anurag Nagar** and **Dr. Anjum Chida** for their guidance and support.

## Contact

For any questions or further information, please contact [Jai Dilbaghi](jai.dilbaghi@gmail.com).

## Project Link

[UTD AI Workshop Final Project](https://github.com/jaithechai/UTD-AI-Internship-Project)
