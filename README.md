AI-Powered Image Colorization

Project Overview
This project is an AI-powered image colorization tool that transforms grayscale images into colored images using a pre-trained deep learning Caffe model. The application is built with Python, OpenCV, PyQt5, and Streamlit, providing a user-friendly interface for both automatic and manual colorization.

Features
- Automatic Colorization: Uses a deep learning model to predict and apply colors to grayscale images.
- Manual Color Enhancement: Allows users to select specific regions and apply custom colors.
- Lasso Tool for Region Selection: Enables precise selection of image areas for colorization.
- Custom Color Picking:
  - Color selection from a palette
  - Screen color picking for accuracy
  - Saving custom colors for reuse
- Blending Effect: Merges user-selected colors with model-predicted colors for natural results.
- User-Friendly Interface: Built with PyQt5 and Streamlit for seamless interaction.

Tech Stack
- Programming Language: Python
- Libraries & Frameworks:
  - OpenCV (Image Processing)
  - NumPy (Data Handling)
  - PyQt5 (Desktop GUI)
  - Streamlit (Web Interface)
  - Caffe Model (Pre-trained for Colorization)

Installation & Setup
1. Clone the Repository
   git clone https://github.com/your-username/image-colorization.git
   cd image-colorization

2. Install Dependencies
   pip install -r requirements.txt

3. Run the Application
   - For GUI (PyQt5):
     python gui.py
   - For Web App (Streamlit):
     streamlit run app.py

How It Works
1. Upload a grayscale image.
2. Select automatic or manual colorization.
3. Use the lasso tool or predefined selection (circle/rectangle) to mark regions.
4. Pick colors from the palette or screen.
5. Apply and blend colors for enhanced results.
6. Download the final colorized image.

Future Enhancements
- Implement a deep learning-based refinement model.
- Support for real-time video colorization.
- Add an AI-driven color suggestion system based on context.

Contribution
Contributions are welcome! Feel free to fork, submit issues, or create pull requests.

Made with love by Gitanjali https://github.com/gitanjali-salvi
