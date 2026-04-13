# 🤟 Real-Time Sign Language Detection System

A deep learning-based system that detects and classifies hand gestures corresponding to sign language words in real time using a webcam. Built on the TensorFlow Object Detection API with a fine-tuned SSD MobileNet V2 model.

---

## 📌 Detected Signs

| ID | Sign |
|----|------|
| 1 | Hello |
| 2 | Yes |
| 3 | No |
| 4 | Thanks |
| 5 | I Love You |

---

## 🗂️ Project Structure

```
RealTimeObjectDetection/
│
├── Tensorflow/
│   ├── models/                        # TensorFlow Model Garden (cloned)
│   ├── scripts/
│   │   └── generate_tfrecord.py       # Script to generate TFRecord files
│   └── workspace/
│       ├── annotations/
│       │   ├── label_map.pbtxt        # Label map for all 5 classes
│       │   ├── train.record           # Training TFRecord
│       │   └── test.record            # Testing TFRecord
│       ├── images/
│       │   ├── CollectedImages/       # Raw images captured via webcam
│       │   │   ├── hello/
│       │   │   ├── yes/
│       │   │   ├── no/
│       │   │   ├── thanks/
│       │   │   └── iloveyou/
│       │   ├── train/                 # Labelled training images + XMLs
│       │   └── test/                  # Labelled testing images + XMLs
│       ├── models/
│       │   └── my_ssd_mobnet/
│       │       ├── pipeline.config    # Model training configuration
│       │       └── ckpt-*/            # Saved model checkpoints
│       └── pre-trained-models/
│           └── ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/
│
├── DataSetCapture.ipynb               # Webcam image collection notebook
└── Tutorial.ipynb                     # Full training & inference pipeline
```

---

## ⚙️ Requirements

- Python 3.9+
- TensorFlow 2.x
- OpenCV (`cv2`)
- TensorFlow Object Detection API
- LabelImg (for image annotation)
- Protocol Buffers (`protobuf`)

Install the core dependencies:

```bash
pip install tensorflow opencv-python protobuf
```

> For the full TensorFlow Object Detection API setup, follow the [official installation guide](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

---

## 🚀 Getting Started

### Step 1 — Collect Images

Run `DataSetCapture.ipynb` to capture webcam images for each sign. The script collects **15 images per label** with a 5-second preparation delay between classes.

```python
labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
number_imgs = 15
```

Images are saved to:
```
Tensorflow/workspace/images/CollectedImages/<label>/
```

### Step 2 — Annotate Images with LabelImg

1. Open [LabelImg](https://github.com/HumanSignal/labelImg) and load the collected images.
2. Draw bounding boxes around each hand gesture.
3. Save annotations as Pascal VOC `.xml` files alongside the images.
4. Split annotated images into `train/` and `test/` folders.

### Step 3 — Generate TFRecords

Run the following from `Tutorial.ipynb` or directly in the terminal:

```bash
python Tensorflow/scripts/generate_tfrecord.py \
    -x Tensorflow/workspace/images/train \
    -l Tensorflow/workspace/annotations/label_map.pbtxt \
    -o Tensorflow/workspace/annotations/train.record

python Tensorflow/scripts/generate_tfrecord.py \
    -x Tensorflow/workspace/images/test \
    -l Tensorflow/workspace/annotations/label_map.pbtxt \
    -o Tensorflow/workspace/annotations/test.record
```

### Step 4 — Configure the Model

The project uses **SSD MobileNet V2 FPNLite 320x320** pretrained on COCO. Key training parameters:

```python
num_classes       = 5
batch_size        = 4
num_train_steps   = 10000
fine_tune_checkpoint_type = "detection"
```

### Step 5 — Train the Model

```bash
python Tensorflow/models/research/object_detection/model_main_tf2.py \
    --model_dir=Tensorflow/workspace/models/my_ssd_mobnet \
    --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config \
    --num_train_steps=10000
```

### Step 6 — Real-Time Detection

Run the real-time detection loop from `Tutorial.ipynb`. The model loads from the latest checkpoint and performs live inference through the webcam:

```python
cap = cv2.VideoCapture(1)  # Change index if needed for your webcam

# Press 'q' to quit the detection window
```

Detections are visualized with bounding boxes and confidence scores overlaid on the live video feed.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Architecture | SSD MobileNet V2 FPNLite |
| Input Resolution | 320 × 320 |
| Pre-training Dataset | COCO 2017 |
| Fine-tuning Method | Transfer Learning |
| Number of Classes | 5 |
| Training Steps | 10,000 |
| Detection Threshold | 0.5 |

---

## 📷 Data Collection Details

- **Source:** Real-time webcam capture
- **Images per class:** 15
- **Total images:** 75 (before train/test split)
- **Annotation tool:** LabelImg (Pascal VOC XML format)
- **Image naming:** UUID-based unique filenames to avoid conflicts

---

## 🔧 Troubleshooting

**Webcam not detected**
Change the `cv2.VideoCapture` index (try `0` or `1`) based on your system setup.

**Low detection accuracy**
Increase `number_imgs` during collection, capture under varied lighting, and increase `num_train_steps` during training.

**TF Object Detection API import errors**
Ensure Protobuf compilation and the API installation are complete. Refer to the [official docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/).

---

## 📄 License

This project is for educational and research purposes.
