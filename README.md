# Hudl AI Tagging System

AI-assisted volleyball tagging system designed for live Hudl dashboard video.

## V1 goal

The first milestone is player identity:

Live Hudl video -> MSS capture -> YOLO -> BoT-SORT -> Jersey OCR -> persistent player memory.

Example:

Track 17 -> #13
Track 23 -> #6
Track 31 -> #4

If OCR temporarily cannot read a jersey, the remembered identity remains attached to the tracker ID.

## Setup

Use Python 3.11 or 3.12.

PowerShell:

python -m venv venv
.\\venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt

Create:

models/

Then copy yolov8n.pt from your existing Hudl_Tagging_Tools repository into models/.

## Run

Open the Hudl match first, then:

python app.py

A screenshot of the desktop appears. Select only the Hudl video/player area and press ENTER.

The AI window shows player bounding boxes, tracker IDs, remembered jersey numbers, jersey confidence and FPS.

Press Q to stop.

Player memory is saved to output/player_memory.json.

## Current scope

Implemented:
- Live MSS screen capture
- Selectable Hudl video ROI
- YOLO player detection
- BoT-SORT tracking
- EasyOCR jersey recognition
- Persistent jersey voting/memory
- Live tracking overlay
- Player memory JSON output

Next:
- Ball tracking
- Appearance re-identification
- Attack detection
- Set detection
- Dig detection
- Block detection
- Serve detection
- Player-event association
- Confidence-based human review
- Automatic Hudl tagging

This version intentionally does not press keys or create Hudl tags automatically. First validate player identity on real Hudl footage.


## Court filtering

At startup V1 now asks for two selections:

1. Hudl video ROI
2. Volleyball court polygon

The court is now selected as a free-form polygon instead of a rectangle, which handles camera perspective and trapezoidal court views much better. Click around the playable court boundary, then press **ENTER**. Press **R** to reset the points.

Only person detections whose **bottom-center foot point** falls inside the calibrated polygon are treated as players. This prevents coaches, referees, substitutes and other people outside the court from receiving tracker IDs, OCR processing or jersey memory.

During tracking, press **C** to recalibrate the court polygon if the Hudl camera zooms, pans or changes its framing. Press **Q** to exit.

A secondary safety limit keeps at most 12 accepted player detections per frame.



## V3: Ball + performance test

The current test version adds:
- COCO sports-ball detection (class 32)
- lightweight ball smoothing/persistence
- ball status overlay
- YOLO inference timing
- CUDA FP16 inference when CUDA is available
- reduced OCR work: confirmed jersey identities are not repeatedly rescanned
- YOLO input-size control

The ball detector is intentionally a first test. If the generic sports-ball class is unreliable on real Hudl footage, the next step is a dedicated volleyball detector trained on Hudl frames.


## V4: Performance + async OCR

V4 moves EasyOCR into a background worker so OCR no longer blocks the main capture/detection/display loop.

The live overlay reports:
- total FPS
- player count
- ball status
- YOLO inference time
- screen capture time
- OCR queue size

YOLO inference is also tuned for the RTX 3050 test target with a smaller input size, FP16 when CUDA is available, and a detection cap.



## V5: Dedicated volleyball detector

The generic COCO sports-ball class has been removed from the player model. Player detection now handles people only.

V5 adds:
- `ball_detector.py` for a dedicated one-class volleyball YOLO model
- `models/volleyball.pt` as the expected trained model
- **B** hotkey to capture clean Hudl frames into `data/ball_frames/`
- `train_volleyball.py` for fine-tuning YOLOv8n
- `data/volleyball_dataset/data.yaml`

### Collect and label training data directly from Hudl

The **B** key is now an interactive volleyball capture tool, so you do not need to manually type YOLO coordinates.

1. Run `python app.py`.
2. When the volleyball is clearly visible, press **B**.
3. Use the **mouse to drag a tight rectangle around the volleyball**.
4. Press **ENTER** or **S** to save.
5. Press **ESC** if the box is wrong and try again.
6. Repeat this for many different ball positions and appearances.

Every capture automatically creates all three:

`data/volleyball_dataset/images/train/*.jpg`
- the complete clean Hudl frame

`data/volleyball_dataset/labels/train/*.txt`
- the YOLO annotation for the box you drew

`data/volleyball_dataset/crops/*.jpg`
- a tight crop showing exactly how the volleyball looks

The label is automatically written as YOLO class `0` (volleyball), with normalized coordinates. This means the mouse box becomes training data automatically.

Collect diverse examples:
- ball near the net
- ball high in the air
- ball near a player
- ball near the boundary
- motion blur
- small/distant ball
- different camera zoom levels
- different lighting
- partial occlusion
- ball at different court locations

Aim for **500–1000 labeled captures** before serious training. Avoid capturing many nearly identical frames.

### Train

Before training, create a validation split from the captured training data (roughly 80% train / 20% validation). Then run `python train_volleyball.py`. The best model is copied to `models/volleyball.pt`.

For a quick first test, you can train directly from the captured `images/train` set, but a proper validation split is recommended.

Restart `python app.py`. V5 automatically loads `models/volleyball.pt` for ball detection.
