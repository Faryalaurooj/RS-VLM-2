#  **RS-VLM: A Semantic-Guided Vision–Language Framework for Robust Small-Object Detection in Remote Sensing Imagery**  
 This repository implements a **research-grade oriented object detection framework** for high-resolution remote sensing imagery.   
   
 The model integrates:  
- Self-Supervised Geometric Structure Module (SGSM)  
- CLIP-guided semantic alignment  
- Oriented bounding box detection head  
- DOTA-v1.0 dataset support  
- End-to-end training and evaluation pipeline  
   
    
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OMQ0AIAwAwZIgBKn1gjJsdGLBABMhuZt+/JaZIyJmAADwi9VP1NMNAABu1AaU4gUeBSGW2wAAAABJRU5ErkJggg==)  
   
**🚀 Key Features**  
**🧠 SGSM (Self-Supervised Geometric Structure Module)**  
- Rotation consistency learning  
- Multi-scale feature fusion  
- Improves robustness for small and dense objects  
   
    
**🌐 CLIP Fusion Module**  
- Vision-language alignment  
- Reduces domain gap in remote sensing imagery  
- Enhances semantic understanding of objects  
   
    
**📦 Oriented Object Detection Head**  
- Predicts rotated bounding boxes: (cx, cy, w, h, θ)  
- Angle-aware loss for stable rotation regression  
- DOTA-compatible detection format
  
**Core Contributions**  
**1. SGSM Module**  
Encodes geometric structure consistency via self-supervised rotation-aware learning.  
**2. CLIP-based Alignment**  
Bridges vision-language domain gap for remote sensing imagery.  
**3. Oriented Detection Head**  
Enables robust rotated object detection for arbitrary-shaped objects.  
   
     
**📊 Full Evaluation Pipeline**  
- [mAP@0.5 evaluation](mailto:mAP@0.5 "mailto:mAP@0.5")  
- Rotated IoU matching  
- Per-class performance reporting  
   
    
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAUBBAwSf8GGLWDWFDY3ixgjcRZhLMNjNHdQYAwF9cq1rV/vUEAIDX7gcRXAQ2s/16gwAAAABJRU5ErkJggg==)  
   
**📁 Project Structure**  
**SGSM-OD/**  
 **  
   ├── models/  
 **  
   │   ├── backbone.py 
 **  
   │   ├── sgsm.py 
 **  
   │   ├── detection_head.py 
 **  
   │   ├── oriented_head.py  
 **  
   │   ├── clip_fusion.py 
 **  
   │   └── detector.py  
 **  
   │**  
 **  
   ├── datasets/ 
 **  
   │   ├── dota.py 
 **  
   │   ├── transforms.py  
 **  
   │**  
 **  
   ├── utils/  
 **  
   │   ├── losses.py  
 **  
   │   ├── metrics.py 
 **  
   │**  
 **  
   ├── configs/ 
 **  
   │   ├── train_sgsm.yaml  
 **  
   │**  
 **  
   ├── train.py 
 **  
   ├── test.py 
 **  
   ├── inference.py  
 **  
   └── README.md 
 **  
  **  
 **  
  **  
   
**📦 Dataset: DOTA 
   
   
 This project uses the **DOTA-v1.0 dataset** for rotated object detection.  
     
   
 Download dataset:  
- [https://captain-whu.github.io/DOTA/dataset.html  
   
    
   
 Expected format:  
   
 datasets/DOTA/  
   
   ├── images/  
   
   └── labels/  
   
 x1 y1 x2 y2 x3 y3 x4 y4 class difficulty  
   
  ](https://captain-whu.github.io/DOTA/dataset.html "https://captain-whu.github.io/DOTA/dataset.html")  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OMQ2AABAAsSNhwgJOUPcjIpnRgQU2QtIq6DIze3UGAMBf3Gu1VcfXEwAAXrseaJEEL8XMiYMAAAAASUVORK5CYII=)  

   
**⚙️ Installation**  
   
git clone https://github.com/Faryalaurooj/RS-VLM-2.git  
 cd RSVLM    
      
 pip install -r requirements.txt    
 Recommended:    
 - Python ≥ 3.8    
 - PyTorch ≥ 2.0    
 - OpenCV    
 - CUDA-enabled GPU   

 
 # **Configuration**    
 Modify training settings in:    
 configs/train_sgsm.yaml    

 
 # **Training**    
 Run training with:    
 python train.py    
 #      
   
# **📊 Evaluation**    
 Evaluate trained model:    
 python test.py    
 Outputs:    
 - mAP@0.5    
 - class-wise performance    
 - precision/recall statistics    
      
# **Inference**    
 Run inference on a single image:    
 python inference.py    
 Output:    
 - result.png with rotated bounding boxes    
   
# ** Method Overview**  

 ### **Pipeline**    
Input Image    
    ↓    
 Backbone CNN    
    ↓    
 SGSM (Self-Supervised Geometric Refinement)    
    ↓    
 CLIP Fusion (Semantic Alignment)    
    ↓    
 Oriented Detection Head    
    ↓    
 Rotated Object Predictions    
 # **Notes**    
 - CLIP module uses a placeholder encoder in default setup.    
 -  Replace with open_clip for best performance.     
 - Rotated IoU is implemented using OpenCV (not GPU optimized).    
 - Decoder is simplified for research prototyping.    
      
   
