# Football Analysis using YOLO

This project employs YOLO (You Only Look Once) object detection to conduct comprehensive analysis of football matches. The goal is to provide detailed insights into player performance, team dynamics, ball possession, and camera movements during a match.

<p><img height="400" width="1000" src="Football-Analysis.gif" alt="demo"></p>

## Installation

1. **Clone the Repository:**

   ```bash
   git clone
   cd football-analysis-using-yolo
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   If the libraries are not installed, then install them separately(preferable).

```bash
pip install <library name>
```

The following libraries are used in this project:

- ultralytics
- numpy
- opencv-python
- roboflow
- pandas
- pickle
- supervision
- shutil
- scikit-learn
- matplotlib
- pillow

## Usage

1. **Data Preparation:**

   - Place your video footage of the football match in the `input` directory.

2. **Running the Analysis:**

   - Execute the main script `python main.py` to initiate the analysis process.
   - The analysis encompasses the following key steps:
     - Object tracking using YOLO for players, referees, and the football.
     - Estimating camera movements to understand viewpoint changes.
     - Calculating player speed, distance traveled, and determining ball possession.
     - Visualizing analysis results on the video frames.

3. **Output:**
   - The annotated and analyzed video will be saved in the `output_videos` directory for review.

## Code Structure

- **`utils.py`**: Contains utility functions for video I/O operations.
- **`trackers.py`**: Implements the YOLO-based object tracker and interpolation techniques.
- **`team_assigner.py`**: Assigns teams to players based on their visual appearance.
- **`player_ball_assigner.py`**: Determines ball possession among players during the match.
- **`camera_movement_estimator.py`**: Estimates camera movements to analyze perspective changes.
- **`view_transformer.py`**: Transforms object positions based on the camera view for accurate analysis.
- **`speed_and_distance_estimator.py`**: Calculates player speeds and distances traveled for performance evaluation.

## Contributing

Contributions, feedback, and suggestions are highly encouraged! Please feel free to open an issue or submit a pull request with any improvements or new features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

Special thanks to the YOLOv8 team and the contributors of the libraries used in this project for their valuable contributions to the field of object detection and analysis in computer vision.

## Note

- Some files like the models used, .pt(PyTorch) files used and the prediction videos are not attached as they could not be pushed.
- For viewing/getting those files please contact me. I'll be very happy to help.
- Also make sure that the main.py file is running in a virtual environment else there might be a few errors appearing very different from the desired output(s).
