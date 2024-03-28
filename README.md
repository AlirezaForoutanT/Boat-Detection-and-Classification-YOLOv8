# Boat-detection-and-Classification-YOLOv8
This project presents a custom-trained YOLOv8 object detection model for detecting boats in satellite images, specifically annotated with data obtained from drone imagery.
Following training, the detected boat regions are processed to determine their real-world _areas_.
The project integrates _morphological operations_, _visualization of results_, and _statistical analysis_, including counts and categorization of different boat types.
Users can manually set thresholds or utilize an automatic approach based on percentiles derived from the boat area distribution. This project serves as a specialized tool for boat detection and analysis in the Lérins Islands area, contributing to applications in maritime surveillance and monitoring.

**To Run:**
Ensure you have Python 3.11 installed. Install the required packages using:

pip install -r requirements.txt

Run run.py to identify boats in regional images. python run.py

Run Results_and_Statistics.py for statistical analysis of boat detection. python Results_and_Statistics.py

To train a new model place annotated images in the specified dataset folder. Ensure drone imagery metadata is available. Run main.py to train the YOLOv8 model.(Make sure the addresses are correctly placed in the config.yaml)

