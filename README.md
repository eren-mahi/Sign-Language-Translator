# Sign-Language-Translator
Sign-Language-Translator is a computer vision and deep learning-based project that recognizes and translates sign language gestures into text using hand tracking and classification models.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project leverages Python, OpenCV, MediaPipe, and TensorFlow/Keras to detect hand gestures and classify them according to sign language alphabets. It is intended as a foundational tool for sign language translation applications, with demo scripts for testing and a modular codebase for extension.

## Features
- Real-time hand tracking and gesture recognition
- Trained deep learning model for sign language classification
- Modular codebase for ease of experimentation and improvement

## File Structure
```
Main/
├── demo/
│   ├── app.py              # Demo GUI application
│   ├── hand_tracking.py    # Demo for hand detection & tracking
├── src/
│   ├── app.py              # Main application logic
│   ├── hand_tracking.py    # Core hand detection module
│   └── models/
│       ├── keras_model.h5  # Trained deep learning model
│       └── labels.txt      # Class labels used by the model
├── requirements.txt        # List of required Python packages
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/eren-mahi/Sign-Language-Translator.git
   cd Sign-Language-Translator/Main
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Run the demo application:**
   ```bash
   python demo/app.py
   ```

2. **Run the main GUI:**
   ```bash
   python src/app.py
   ```

3. **Model files:**
   - Ensure `keras_model.h5` and `labels.txt` are present in `src/models/`.

4. **Hand tracking test:**
   ```bash
   python src/hand_tracking.py
   ```

## Requirements
- Python 3.6+
- TensorFlow
- NumPy
- OpenCV-Python
- MediaPipe
- Pillow

Refer to `requirements.txt` for a full list.

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
Feel free to use, modify, and share under the repository's license, or specify a license in the project if missing.

---

**Note:** If you use this code for research or commercial purposes, please cite the repository and respect the original author's work.
