import cv2
import numpy as np
import math

# =============================================================================
# HELPER AND UI FUNCTIONS
# =============================================================================

def make_odd(value, min_val=1):
    """
    Kernel sizes (3x3, 5x5, etc.) must always be odd numbers.
    If an even number is passed, it adds 1 to make it odd.
    """
    value = int(value)
    if value % 2 == 0:
        value += 1
    return max(min_val, value)

def draw_hud(img, title, info_lines):
    """
    Draws a professional Heads-Up Display (HUD) overlay to prevent text overlap
    and ensure readability on any background color.
    """
    h, w = img.shape[:2]
    
    # Panel settings
    panel_w = 350  # Width of the info panel
    panel_h = 40 + (len(info_lines) * 25) # Height based on number of lines
    margin = 10
    
    # Semi-transparent background (Overlay)
    overlay = img.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + panel_w, margin + panel_h), (0, 0, 0), -1)
    
    # Blend the overlay with the image (Alpha blending)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw border
    cv2.rectangle(img, (margin, margin), (margin + panel_w, margin + panel_h), (255, 255, 255), 1)

    # Title Text
    cv2.putText(img, title, (margin + 10, margin + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Detail Lines
    y_start = margin + 55
    for i, line in enumerate(info_lines):
        y_pos = y_start + (i * 25)
        # 1. Black shadow (for readability)
        cv2.putText(img, line, (margin + 11, y_pos + 1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        # 2. White text
        cv2.putText(img, line, (margin + 10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

def nothing(x):
    pass

# =============================================================================
# PROCESSING LOGIC: FILTER GROUPS
# =============================================================================

def process_image(img, category, sub_mode, p1, p2):
    """
    Main logic where all image processing algorithms branch out.
    Returns: (Processed Image, Info List)
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result = img.copy()
    info = []
    
    # -----------------------------------------------------------------
    # CATEGORY 0: BLURRING & SMOOTHING
    # -----------------------------------------------------------------
    if category == 0:
        k_size = make_odd(p1) # Kernel size
        
        if sub_mode == 0: # Average Blur
            result = cv2.blur(img, (k_size, k_size))
            info = ["Type: Average (Box) Blur", f"Kernel: {k_size}x{k_size}", "Simple averaging."]
            
        elif sub_mode == 1: # Gaussian Blur
            sigma = p2 / 10.0
            result = cv2.GaussianBlur(img, (k_size, k_size), sigma)
            info = ["Type: Gaussian Blur", f"Kernel: {k_size}x{k_size}", f"Sigma: {sigma:.1f}"]
            
        elif sub_mode == 2: # Median Blur
            k_med = min(k_size, 55) # Avoid extreme slowness
            result = cv2.medianBlur(img, k_med)
            info = ["Type: Median Blur", f"Kernel: {k_med}", "Best for Salt-and-Pepper noise."]
            
        elif sub_mode == 3: # Bilateral Filter
            d = max(1, int(p1 / 2)) # Diameter
            sigma_color = p2
            sigma_space = p2
            result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
            info = ["Type: Bilateral Filter", f"Diameter: {d}", f"Sigma: {sigma_color}", "Preserves edges, smooths surface."]

    # -----------------------------------------------------------------
    # CATEGORY 1: EDGE DETECTION
    # -----------------------------------------------------------------
    elif category == 1:
        k_size = make_odd(p1, min_val=3)
        k_size = min(k_size, 7) # Max 7 recommended for Sobel
        
        if sub_mode == 0: # Canny
            low = p1 * 2
            high = p2 * 2
            edges = cv2.Canny(img_gray, low, high)
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            info = ["Type: Canny Edge Detector", f"Lower Thresh: {low}", f"Upper Thresh: {high}"]
            
        elif sub_mode == 1: # Sobel
            dx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=k_size)
            dy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=k_size)
            alpha = p2 / 255.0
            comb = cv2.addWeighted(cv2.convertScaleAbs(dx), alpha, cv2.convertScaleAbs(dy), 1-alpha, 0)
            result = cv2.cvtColor(comb, cv2.COLOR_GRAY2BGR)
            info = ["Type: Sobel", f"Kernel: {k_size}", f"X/Y Balance: %{int(alpha*100)}"]
            
        elif sub_mode == 2: # Laplacian
            lap = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=k_size)
            gain = max(1, p2 / 20.0)
            result_gray = cv2.convertScaleAbs(cv2.multiply(lap, gain))
            result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
            info = ["Type: Laplacian (2nd Derivative)", f"Kernel: {k_size}", f"Brightness Gain: {gain:.1f}x"]
            
        elif sub_mode == 3: # Prewitt (Manual)
            kx = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
            ky = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
            px = cv2.filter2D(img_gray, -1, kx)
            py = cv2.filter2D(img_gray, -1, ky)
            result_gray = cv2.addWeighted(px, 0.5, py, 0.5, 0)
            result = cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
            info = ["Type: Prewitt (Manual Kernel)", "Kernel: 3x3 Fixed", "Sum of Vertical & Horizontal Gradients"]

        elif sub_mode == 4: # Scharr
            dx = cv2.Scharr(img_gray, cv2.CV_64F, 1, 0)
            dy = cv2.Scharr(img_gray, cv2.CV_64F, 0, 1)
            comb = cv2.addWeighted(cv2.convertScaleAbs(dx), 0.5, cv2.convertScaleAbs(dy), 0.5, 0)
            result = cv2.cvtColor(comb, cv2.COLOR_GRAY2BGR)
            info = ["Type: Scharr Filter", "Special Kernel", "More sensitive/accurate than Sobel."]

    # -----------------------------------------------------------------
    # CATEGORY 2: THRESHOLDING
    # -----------------------------------------------------------------
    elif category == 2:
        if sub_mode == 0: # Simple Binary
            val = p2
            _, th = cv2.threshold(img_gray, val, 255, cv2.THRESH_BINARY)
            result = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            info = ["Type: Binary Threshold", f"Cutoff: {val}", "Pixel > Thresh = White, else Black"]
            
        elif sub_mode == 1: # Otsu
            val, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            result = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            info = ["Type: Otsu Thresholding", f"Calc. Thresh: {val}", "Auto-calculates optimal threshold."]
            
        elif sub_mode == 2: # Adaptive Mean
            block = make_odd(p1, min_val=3)
            C = (p2 / 10.0) - 10
            th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C)
            result = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            info = ["Type: Adaptive (Mean)", f"Block: {block}", f"Constant C: {C:.1f} (Shadow tolerance)"]
            
        elif sub_mode == 3: # Adaptive Gaussian
            block = make_odd(p1, min_val=3)
            C = (p2 / 10.0) - 10
            th = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)
            result = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
            info = ["Type: Adaptive (Gaussian)", f"Block: {block}", f"Constant C: {C:.1f}", "Ideal for scanning documents."]

    # -----------------------------------------------------------------
    # CATEGORY 3: MORPHOLOGICAL OPERATIONS
    # -----------------------------------------------------------------
    elif category == 3:
        k_size = make_odd(p1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        
        op = None
        name = ""
        desc = ""
        
        mod_index = sub_mode % 7
        
        if mod_index == 0:
            op = cv2.MORPH_ERODE
            name = "Erosion"
            desc = "Thins white regions."
        elif mod_index == 1:
            op = cv2.MORPH_DILATE
            name = "Dilation"
            desc = "Thickens white regions."
        elif mod_index == 2:
            op = cv2.MORPH_OPEN
            name = "Opening"
            desc = "Removes small noise (Erosion -> Dilation)."
        elif mod_index == 3:
            op = cv2.MORPH_CLOSE
            name = "Closing"
            desc = "Closes small holes (Dilation -> Erosion)."
        elif mod_index == 4:
            op = cv2.MORPH_GRADIENT
            name = "Morph Gradient"
            desc = "Outline of the object."
        elif mod_index == 5:
            op = cv2.MORPH_TOPHAT
            name = "Top Hat"
            desc = "Extracts bright details."
        elif mod_index == 6:
            op = cv2.MORPH_BLACKHAT
            name = "Black Hat"
            desc = "Extracts dark details."

        result = cv2.morphologyEx(img, op, kernel, iterations=1)
        info = [f"Type: {name}", f"Kernel: {k_size}x{k_size}", desc]

    # -----------------------------------------------------------------
    # CATEGORY 4: COLOR & CONTRAST
    # -----------------------------------------------------------------
    elif category == 4:
        if sub_mode == 0: # Histogram Equalization
            yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
            result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
            info = ["Type: Histogram Equalization", "Global", "Spreads contrast across the image."]
            
        elif sub_mode == 1: # CLAHE
            clip = max(1.0, p2 / 10.0)
            grid = max(1, int(p1 / 4))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            info = ["Type: CLAHE (Adaptive)", f"Clip Limit: {clip:.1f}", f"Grid: {grid}x{grid}"]
            
        elif sub_mode == 2: # Gamma Correction
            gamma = p2 / 50.0 + 0.1 
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            result = cv2.LUT(img, table)
            info = ["Type: Gamma Correction", f"Gamma: {gamma:.2f}", "Low gamma: Brighten, High gamma: Darken"]
            
        elif sub_mode == 3: # Invert (Negative)
            result = cv2.bitwise_not(img)
            info = ["Type: Negative (Invert)", "Inverted Colors", "255 - Pixel value"]
            
        elif sub_mode == 4: # Sharpen
            blur = cv2.GaussianBlur(img, (0,0), 3)
            amount = p2 / 50.0
            result = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
            info = ["Type: Unsharp Masking (Sharpen)", f"Amount: {amount:.1f}", "Subtracts blur to sharpen details."]

    # -----------------------------------------------------------------
    # CATEGORY 5: FEATURE DETECTION
    # -----------------------------------------------------------------
    elif category == 5:
        if sub_mode == 0: # Harris Corner
            gray_float = np.float32(img_gray)
            bs = make_odd(p1 / 5, min_val=2)
            k = 0.04 + (p2 / 1000.0)
            dst = cv2.cornerHarris(gray_float, bs, 3, k)
            dst = cv2.dilate(dst, None)
            result = img.copy()
            result[dst > 0.01 * dst.max()] = [0, 0, 255]
            info = ["Type: Harris Corner", f"Block: {bs}", "Red dots indicate corners."]
            
        elif sub_mode == 1: # Shi-Tomasi
            max_corners = int(p1 * 2) + 10
            quality = max(0.01, p2 / 100.0)
            corners = cv2.goodFeaturesToTrack(img_gray, max_corners, quality, 10)
            result = img.copy()
            if corners is not None:
                corners = np.int32(corners)
                for i in corners:
                    x, y = i.ravel()
                    cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
            info = ["Type: Shi-Tomasi", f"Max Corners: {max_corners}", f"Quality: {quality:.2f}", "Green dots are strong corners."]
            
        elif sub_mode == 2: # FAST Detector
            threshold = int(p2 / 2) + 10
            fast = cv2.FastFeatureDetector_create(threshold=threshold)
            kp = fast.detect(img, None)
            result = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
            info = ["Type: FAST Feature Detector", f"Threshold: {threshold}", "Blue rings are detected keypoints."]

    # Fallback if category or mode is undefined
    if not info:
        info = ["MOD: Undefined", "Mode not found in this category.", "Please slide the SUB-MODE bar."]

    return result, info

# =============================================================================
# MAIN PROGRAM
# =============================================================================

def main():
    print("Program starting...")
    
    # -------------------------------------------------------------------
    # PLEASE ENTER YOUR IMAGE PATH HERE
    file_path = "bobRoss.jpg"
    # -------------------------------------------------------------------
    
    img_original = cv2.imread(file_path)
    if img_original is None:
        print(f"ERROR: Image not found at '{file_path}'.")
        return

    # Resize to fit screen (Max Width 800px)
    target_w = 800
    h, w = img_original.shape[:2]
    scale = target_w / w
    new_h = int(h * scale)
    img = cv2.resize(img_original, (target_w, new_h))

    WINDOW_NAME = "Ultimate Image Lab v3.1 (English Version)"
    cv2.namedWindow(WINDOW_NAME)

    # Create Trackbars
    cv2.createTrackbar("CATEGORY", WINDOW_NAME, 0, 5, nothing)
    cv2.createTrackbar("SUB-MODE", WINDOW_NAME, 0, 6, nothing)
    cv2.createTrackbar("PARAM 1", WINDOW_NAME, 5, 100, nothing)
    cv2.createTrackbar("PARAM 2", WINDOW_NAME, 127, 255, nothing)

    print("\n--- CONTROL PANEL ---")
    print("0: Blur/Smoothing")
    print("1: Edge Detection")
    print("2: Thresholding")
    print("3: Morphology")
    print("4: Color/Contrast")
    print("5: Feature Detection")
    print("---------------------")
    print("Press 'q' or 'ESC' to exit.")

    while True:
        # Read Trackbar Values
        category = cv2.getTrackbarPos("CATEGORY", WINDOW_NAME)
        sub_mode = cv2.getTrackbarPos("SUB-MODE", WINDOW_NAME)
        p1 = cv2.getTrackbarPos("PARAM 1", WINDOW_NAME)
        p2 = cv2.getTrackbarPos("PARAM 2", WINDOW_NAME)
        
        try:
            processed_img, info_list = process_image(img, category, sub_mode, p1, p2)
        except Exception as e:
            processed_img = img.copy()
            info_list = ["ERROR OCCURRED!", str(e)[:40], "Try changing parameters."]
        
        # Display "ORIGINAL" text on the left image
        img_display = img.copy()
        cv2.putText(img_display, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Draw HUD on the processed image
        title = f"CAT: {category} | MODE: {sub_mode}"
        draw_hud(processed_img, title, info_list)
        
        # Stack images horizontally
        final_output = np.hstack((img_display, processed_img))
        cv2.imshow(WINDOW_NAME, final_output)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27: # q or ESC
            break

    cv2.destroyAllWindows()
    print("Program terminated.")

if __name__ == "__main__":
    main()