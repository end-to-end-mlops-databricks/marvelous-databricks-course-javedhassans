<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture. 
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset. 
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the 25th of November.


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11. 
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv venv -p 3.11.0 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```



Here’s an enhanced explanation of your dataset based on the information from `data_dictionary.csv`, covering each instrument's purpose and the fields it includes:

### 1. **Identifier**
   - **`id`**: The unique identifier assigned to each participant, which is used to match records across different files and data sources.

### 2. **Demographics**
   - **`Basic_Demos-Enroll_Season`**: The season during which a participant enrolled in the study, which may help in analyzing seasonal trends or impacts.
   - **`Basic_Demos-Age`**: The participant’s age, likely a key demographic feature.
   - **`Basic_Demos-Sex`**: Gender of the participant, encoded as `0` for Male and `1` for Female.

### 3. **Internet Use and Educational History**
   - **`PreInt_EduHx-computerinternet_hoursday`**: Measures daily internet/computer usage hours before any intervention. This could provide a baseline for understanding internet dependency.
   - **`Parent-Child Internet Addiction Test (PCIAT)`**: Includes **`PCIAT-PCIAT_Total`**, a total score measuring the severity of internet addiction (compulsivity, escapism, and dependency). This score is pivotal as the **target variable `sii`** is derived from it, categorizing internet addiction into four levels:
      - `0`: None
      - `1`: Mild
      - `2`: Moderate
      - `3`: Severe

### 4. **Children's Global Assessment Scale (CGAS)**
   - **`CGAS-Season`**: Season when the assessment was conducted.
   - **`CGAS-CGAS_Score`**: A numerical scale used by mental health clinicians to assess general functionality in youth, with higher scores indicating better functioning.

### 5. **Physical Measures**
   - **`Physical-Season`**: The season of data collection, which could affect measures like weight or blood pressure.
   - **`Physical-BMI`, `Physical-Height`, `Physical-Weight`, `Physical-Waist_Circumference`**: These biometric indicators measure aspects of the participant's physical health.
   - **`Physical-Diastolic_BP`, `Physical-HeartRate`, `Physical-Systolic_BP`**: Blood pressure and heart rate measurements are vital for understanding cardiovascular health.

### 6. **FitnessGram and Treadmill Data**
   - **FitnessGram Vitals and Treadmill**: Cardiovascular fitness assessments, likely involving treadmill-based tests to evaluate endurance and physical capacity.
   - **FitnessGram Child**: Measures various aspects of physical fitness, including:
      - **Aerobic capacity**, **muscular strength**, **muscular endurance**, **flexibility**, and **body composition**.
      - These fields help assess the participant's overall fitness and physical health, relevant for understanding correlations with internet use or sleep quality.

### 7. **Bio-electric Impedance Analysis (BIA)**
   - Provides in-depth body composition data, including:
      - **BMI**, **body fat percentage**, **lean muscle mass**, and **water content**.
   - These measurements are essential for a comprehensive view of physical health and can be related to other health metrics, such as sleep or mental well-being.

### 8. **Physical Activity Questionnaire (PAQ)**
   - **`PAQ_A` and `PAQ_C`**: Both assess the participant’s physical activity level over the last week, specifically focusing on vigorous activities. This is relevant for gauging overall physical engagement and comparing it with sedentary behaviors like internet use.

### 9. **Sleep Disturbance Scale (SDS)**
   - Designed to categorize sleep disorders in children, this scale includes **Sleep Disturbance Scores** that could help in analyzing the relationship between sleep quality and variables like screen time or physical fitness.

### 10. **Actigraphy Data**
   - **Accelerometer Data**: Includes continuous measurements for up to 30 days, capturing data on physical movement and activity trends in natural settings.
     - **X, Y, Z axes**: Measure acceleration along each axis to capture movement intensity.
     - **ENMO**: Calculates net motion, where zero indicates inactivity, which could correspond to periods of sleep or rest.
     - **Angle-Z**: Measures the angle of the arm relative to a horizontal plane, which could help in detecting activity types.
     - **Non-wear flag**: Identifies when the accelerometer wasn’t worn, aiding in filtering out non-activity data.
     - **Ambient Light, Battery Voltage, Time of Day, Weekday, Quarter, Relative Date**: Provides contextual data that can be used to understand behavioral and temporal patterns.

### Summary of Data Utility
This dataset provides a holistic view of each participant’s demographic, physical, mental, and behavioral characteristics. By combining data on internet use, sleep disturbance, physical fitness, body composition, and actigraphy, the study is positioned to explore the relationships between sedentary behaviors, physical health, mental well-being, and potential internet addiction.

This setup could support various analyses:
1. **Predicting Internet Addiction Levels**: Using `PCIAT` scores and demographic/health data.
2. **Correlating Physical Activity with Internet Use or Sleep**: Using actigraphy and PAQ data.
3. **Analyzing Sleep and Health Relationships**: Leveraging SDS data with physical and mental health scores.
