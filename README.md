Sight2Text is a self-project that demonstrates how Vision Transformers (SigLIP) can be combined with a Text Decoder (Gemma) to build an image-to-text system. The project implements a custom pipeline for preprocessing, tokenization, and model weight management, followed by an end-to-end inference workflow for vision-to-language tasks.

Tech Stack:
Programming Language: Python
Frameworks & Libraries: PyTorch, Hugging Face Transformers
Data Processing: NumPy, Pillow
Model Handling: Safetensors
Utilities: TQDM, Fire

Project Structure:
├── inference.py              # Runs end-to-end inference  
├── launch_inference.sh       # Shell script to launch inference  
├── modeling_siglip.py        # SigLIP Vision Transformer model  
├── modeling_gemma.py         # Gemma Text Decoder model  
├── processing_paligemma.py   # Preprocessing & tokenization pipeline  
├── utils.py                  # Helper functions and utilities  
├── requirements.txt          # Project dependencies  
├── test/pic1.jpeg            # contain test image
