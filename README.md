# Eye Tracking Project based on MobileGaze
--- 
<br>

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download weight files:

   a) Download weights from the following links:

   | Model        | PyTorch Weights                                                                                             | ONNX Weights                                                                                                        | Size    | Epochs | MAE*  |
   | ------------ | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------- | ------ | ----- |
   | ResNet-18    | [resnet18.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet18.pt)               | [resnet18_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet18_gaze.onnx)         | 43 MB   | 200    | 12.84 |
   | ResNet-34    | [resnet34.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet34.pt)               | [resnet34_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet34_gaze.onnx)         | 81.6 MB | 200    | 11.33 |
   | ResNet-50    | [resnet50.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet50.pt)               | [resnet50_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/resnet50_gaze.onnx)         | 91.3 MB | 200    | 11.34 |
   | MobileNet V2 | [mobilenetv2.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobilenetv2.pt)         | [mobilenetv2_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobilenetv2_gaze.onnx)   | 9.59 MB | 200    | 13.07 |
   | MobileOne S0 | [mobileone_s0_fused.pt](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobileone_s0.pt) | [mobileone_s0_gaze.onnx](https://github.com/yakhyo/gaze-estimation/releases/download/v0.0.1/mobileone_s0_gaze.onnx) | 4.8 MB  | 200    | 12.58 |
   | MobileOne S1 | [not available](#)                                                                                          | [not available](#)                                                                                                  | xx MB   | 200    | \*    |
   | MobileOne S2 | [not available](#)                                                                                          | [not available](#)                                                                                                  | xx MB   | 200    | \*    |
   | MobileOne S3 | [not available](#)                                                                                          | [not available](#)                                                                                                  | xx MB   | 200    | \*    |
   | MobileOne S4 | [not availablet](#)                                                                                         | [not available](#)                                                                                                  | xx MB   | 200    | \*    |

   '\*' - soon will be uploaded (due to limited computing resources I cannot publish rest of the weights, but you still can train them with given code).
   
   *MAE (Mean Absolute Error) - lower values indicate better accuracy in degrees.

   b) Run the command below to download weights to the `weights` directory (Linux):

   ```bash
   # Download specific model weights
   sh download.sh [model_name]
   # Available models: resnet18, resnet34, resnet50, mobilenetv2, mobileone_s0
   
   # Example:
   sh download.sh resnet18
   ```
---
<br>

## For the demo, please check the [demo.ipynb](demo.ipynb)

## For the original project, please check the [gaze-estimation](https://github.com/yakhyo/gaze-estimation) repository.

## For batch processing the video data, please fill in these files and run them in orders:

### 1. Inference the gaze 
  ```bash
    python inference.py 
    --model resnet18 
    --weight weights/resnet18.pt 
    --view --source [source_video] 
    --output [output_file] 
  ```
### 2. Extract the calibration data from the inference output
  ```bash
  python batch_run_cal.py
  ```
### 3. Get the Prediction Polynomial Coefficients
  ```bash
  python batch_run_poly.py
  ```
### 4. Predict the other gaze data
  ```bash
  python batch_run_predict.py
  ```
### 5. Plot the heatmap
  ```bash
  python batch_run_kde.py
  ```

