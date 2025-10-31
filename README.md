# CAVAS
Simulation of a computer-vision–based intelligent vehicle alert system that plays adaptive engine sounds for pedestrian and vehicle detection.
# CAVAS - Context-Aware Acoustic Vehicle Alerting System (Simulation)

Short: YOLOv8-powered system that detects pedestrians, vehicles, and animals and simulates context-aware engine sound playback.

## Features
- Real-time object detection (YOLOv8) from webcam
- Filters detections to relevant classes (person, vehicle, animal)
- Mock sensor confirmation step (replaceable with Arduino serial)
- Plays engine sound locally when detections are confirmed
- Logs detections to `logs/detections.csv`
- Simple analysis script `analyze_logs.py` to visualize detection counts

## Quick start
1. Clone repo
2. Create virtualenv and install requirements:
<p align="center">
  <img src="images (1).jpg" width="600" alt="Detection Demo">
</p>
<p align="center">
  <img src="download (2).jpg" width="600" alt="Detection Demo">
</p>


   
