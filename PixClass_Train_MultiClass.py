# Script for classifying pixels as vegetation vs. multiple background classes.
# The script uses annotated photos (labelme) to train a RandomForest classification model.
# Instructions: Add training folder (.json), validation folder (unbiased images), batch folder, and output folder. Tweak model features as needed.
# Miles Innes. February 24, 2026. Script created with the help of Google AI.

### Imports ############################################################################################################
import os, cv2, json, glob
import numpy as np
import pandas as pd
import random
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, r2_score, confusion_matrix
from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from skimage.feature import hessian_matrix, hessian_matrix_eigvals, local_binary_pattern
from skimage.color import rgb2lab, lab2lch
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.filters.rank import entropy, modal
from skimage.filters import scharr, laplace, unsharp_mask
from skimage.morphology import disk, erosion, dilation
from tqdm import tqdm


### Define Directories #################################################################################################
### Image-Level Split Validation (60 images for training, 15 for testing ONLY) ###
# Training Images (60 Annotated)
#TRAIN_DIR = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\High Resolution\Combined_Train_MR"
# Validation Images (15 Annotated)
#VAL_DIR = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\High Resolution\Combined_Validation_MR"

### 5-Fold Validation Testing (Train on all 75 images with 5-fold split, less robust against spatial autocorrelation ###
# AP Photos Input
#AP_INPUT = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\Combined_AP_Input"
#AP_INPUT = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\High Resolution\AP_In_Test"
#AP_INPUT = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\AP_In_Test"

# AP Photos Output
#AP_OUTPUT = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\ComE"
#AP_OUTPUT = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\High Resolution\AP_Out_Test"
#AP_OUTPUT = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\AP_Out_Test"
#TEST_DIR = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\High Resolution\Combined_5Fold_MR"
TEST_DIR = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Field Work\Photos\Combined_5Fold"
EXPORT_DIR = r"C:\Users\milesinn\Desktop\Miles Innes\Projects\MS Thesis\Figures\Paper"

# Set seeds
np.random.seed(99)
random.seed(99)


### Assign Categorical Features (Burn Status & Sampling Date) *** UNUSED *** ###########################################
BURNED_UNITS = ["BCE", "BCW", "BH", "BL", "EC", "JW", "KC", "MM", "UE"]
UNBURNED_UNITS = ["BHU", "BTU", "ECU", "KCU", "SC"]
UNIT_MAP = {unit: i for i, unit in enumerate(['BCE', 'BCW', 'BH', 'BL', 'EC', 'JW', 'KC', 'MM', 'UE', 'BHU', 'BTU', 'ECU', 'KCU', 'SC'])}

# Extract Unit ID / Sampling Date / Burn Status from image name
def parse_metadata(filename):
    # 1. Extract Unit
    unit_str = filename.split('_')[0]
    unit_id = UNIT_MAP.get(unit_str, -1)  # Returns -1 if unit isn't in our list
    is_burned = 1 if unit_str in BURNED_UNITS else 0

    # 2. Extract Date
    date_str = filename.split('_')[3].split('.')[0]
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    day = date_obj.timetuple().tm_yday  # 1-365

    #start_date = datetime(date_obj.year, 6, 25)
    #days_since_start = (date_obj - start_date).days

    return is_burned, day  # Now returning 3 values


### Prepare Feature Extraction #########################################################################################
# Define JSON classes from annotations
CLASS_MAP = {
    "_background_": 0, # Unannotated Regions
    "Vegetation": 1,
    "Mineral Soil": 2,
    "Litter": 3,
    "Wood": 4, # Merge as 'Coarse Substrate'
    "Rock": 4, # Merge as 'Coarse Substrate'
    "Quadrat": 5,
    "Wood Char": 4, # Merge as 'Coarse Substrate'
    "Senesced Vegetation": 1  # Merge late-stage annotations with early-stage annotations
}

# The list index must match the Class Map (above)
CLASS_NAMES = [
    "Non-Annotated",    # 0
    "Vegetation",       # 1
    "Mineral Soil",     # 2
    "Litter",           # 3
    "Coarse Substrate", # 4
    "Quadrat"           # 5
    ]

# Define class colors (for prediction mapping)
CLASS_COLORS = {
    0: (0, 0, 0),
    1: (0, 255, 0),
    2: (19, 69, 139),
    3: (0, 255, 255),
    4: (128, 128, 128),
    5: (255, 255, 255)
    #6: (128, 128, 128)
    #7: (128, 128, 128)
}

# Define line thickness (px) i.e. buffer around line annotations
LINE_THICKNESS = 2


### Modelling Framework (5 Fold Validation) ############################################################################
'''
def run_random_5fold(X, y):
    """
    X: Feature matrix (e.g., shape [500000, 14])
    y: Stacked labels (e.g., shape [500000])
    """
    # Initialize KFold with Shuffling
    kf = KFold(n_splits=5, shuffle=True, random_state=99)

    fold_accuracies = []
    fold_r2 = []

    print(f"Starting Pixel-Level Randomized 5-Fold CV...")
    print(f"Total Pixels: {len(y)}")
    print("-" * 30)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # Split into 80% Train, 20% Validation
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Initialize the Model
        rf = RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1
        )

        # Train
        rf.fit(X_train, y_train)

        # Predict
        y_pred = rf.predict(X_val)

        # Calculate Metrics
        acc = accuracy_score(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        fold_accuracies.append(acc)
        fold_r2.append(r2)

        print(f"Fold {fold + 1} | Accuracy: {acc:.4f} | R2: {r2:.4f}")

    # --- Final Summary ---
    print("-" * 30)
    print("PIXEL-LEVEL MODEL PERFORMANCE")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean R-Squared: {np.mean(fold_r2):.4f}")
    print("-" * 30)

    # Return the last model and the final classification report
    print("Final Fold Detailed Report:")
    presentClasses = np.unique(y_val)
    target_names = [CLASS_NAMES[i] for i in presentClasses]
    print(classification_report(y_val, y_pred, target_names=target_names))

    return rf
    '''

##############################
# RF (WITH GRAPHS)
def run_random_5fold(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=99)

    fold_accuracies = []
    fold_r2 = []
    all_importances = []

    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    combined_conf_matrix = np.zeros((num_classes, num_classes))

    print(f"Starting Pixel-Level Randomized 5-Fold CV...")
    print(f"Total Pixels: {len(y)}")
    print("-" * 30)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        rf = RandomForestClassifier(
            n_estimators=150,
            n_jobs=-1,
            random_state=99
        )

        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        fold_accuracies.append(acc)
        fold_r2.append(r2)
        all_importances.append(rf.feature_importances_)

        fold_cm = confusion_matrix(y_val, y_pred, labels=unique_classes)
        combined_conf_matrix += fold_cm

        print(f"Fold {fold + 1} | Accuracy: {acc:.4f} | R2: {r2:.4f}")

    print("-" * 30)
    print("PIXEL-LEVEL MODEL PERFORMANCE")
    print(f"Mean Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean R-Squared: {np.mean(fold_r2):.4f}")
    print("-" * 30)

    print("Final Fold Detailed Report:")
    target_names = [CLASS_NAMES[i] for i in unique_classes]
    print(classification_report(y_val, y_pred, target_names=target_names))

    # Ensure export directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)

    # Global ggplot-style font settings (Arial/Helvetica matching R)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans']
    plt.rcParams['text.color'] = '#333333'
    plt.rcParams['axes.labelcolor'] = '#000000'
    plt.rcParams['xtick.color'] = '#333333'
    plt.rcParams['ytick.color'] = '#333333'

    # ==========================================
    # 1. GENERATE MEAN FEATURE IMPORTANCE PLOT (theme_bw style)
    # ==========================================
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances,
        'Std': std_importances
    }).sort_values(by='Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor('white')
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(0.8)

    ax.grid(True, color='#E5E5E5', linestyle='-', linewidth=0.8, which='major')

    ax.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], xerr=feat_imp_df['Std'],
            color='#56B4E9', alpha=0.9, ecolor='#333333', capsize=4, height=0.65)

    ax.set_xlabel('Mean Decrease in Impurity (MDI)', fontsize=11, labelpad=10)
    ax.set_title('Random Forest Feature Importance (5-Fold Average)', fontsize=13, pad=15, loc='left',
                 fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10, length=4, color='#333333')

    plt.tight_layout()
    importance_path = os.path.join(EXPORT_DIR, 'rf_feature_importance.png')
    plt.savefig(importance_path, dpi=300)
    plt.close()
    print(f"Saved: {importance_path}")

    # ==========================================
    # 2. GENERATE CONFUSION MATRIX HEATMAP (theme_bw style)
    # ==========================================
    cm_normalized = combined_conf_matrix.astype('float') / combined_conf_matrix.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#333333')
        spine.set_linewidth(0.8)

    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar_kws={'label': 'Proportion of Pixels'},
        ax=ax,
        annot_kws={'fontsize': 10, 'color': '#000000'}
    )

    ax.set_ylabel('True Class', fontsize=11, labelpad=10, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=11, labelpad=10, fontweight='bold')
    ax.set_title('Normalized Confusion Matrix (5-Fold Accumulated)', fontsize=13, pad=15, loc='left', fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=10, length=4, color='#333333')

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10, length=4, color='#333333')
    cbar.ax.yaxis.label.set_size(11)
    cbar.outline.set_visible(True)
    cbar.outline.set_edgecolor('#333333')
    cbar.outline.set_linewidth(0.8)

    plt.tight_layout()
    cm_path = os.path.join(EXPORT_DIR, 'rf_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved: {cm_path}")

    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_importances,
        'Std': std_importances
    })
    importance_df.to_csv(os.path.join(EXPORT_DIR, 'raw_importances.csv'), index=False)

    # Save Normalized Confusion Matrix
    cm_normalized = combined_conf_matrix.astype('float') / combined_conf_matrix.sum(axis=1)[:, np.newaxis]
    cm_df = pd.DataFrame(cm_normalized, index=target_names, columns=target_names)
    cm_df.to_csv(os.path.join(EXPORT_DIR, 'raw_confusion_matrix.csv'))

    return rf


### Model Feature Extraction ###########################################################################################
# Define Model Features
def extract_features(img):

    ### --- Categorical Features (Unused) --- ####################################
    # h, w = img.shape[:2]
    # burn_feat = np.full((h, w), burned, dtype=np.float32)
    # unit_feat = np.full((h, w), unit, dtype=np.float32)
    # day_feat = np.full((h, w), day, dtype=np.float32)


    ### --- CLAHE SETUP --- ###########################################################
    #clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(12, 12))

    #img = (img - img.mean()) / img.std()


    ### Color Features #################################################################################################
    ##### HSV (Unused)
    #hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    #H, S, V = hsv[:,:,0], hsv[:,:,1], hsv[:,:,2]
    #H_rad = H * (np.pi / 90.0)
    #H_sin = np.sin(H_rad)
    #H_cos = np.cos(H_rad)


    ##### LAB (Primary Color Space)
    #lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab = rgb2lab(img).astype(np.float32)
    #lch = lab2lch(lab).astype(np.float32)
    #L, C, H = lch[:, :, 0], lch[:, :, 1], lch[:, :, 2]
    L, A, B = lab[:,:,0], lab[:,:,1], lab[:,:,2]
    Ln = (L - L.mean()) / L.std()

    #Chroma = np.sqrt((A) ** 2 + (B) ** 2)
    #L_CLAHE = clahe.apply(L.astype(np.uint8)).astype(np.float32)

    # LAB Bilateral Blurs
    #A_bilateral = cv2.bilateralFilter(A, d=15, sigmaColor=20, sigmaSpace=15)
    #B_bilateral = cv2.bilateralFilter(B, d=15, sigmaColor=20, sigmaSpace=15)
    L_B3 = cv2.bilateralFilter(L, d=0, sigmaColor=25, sigmaSpace=3)
    L_B6 = cv2.bilateralFilter(L, d=0, sigmaColor=25, sigmaSpace=6)
    L_B9 = cv2.bilateralFilter(L, d=0, sigmaColor=25, sigmaSpace=9)

    # Gaussian Blurs
    #L_GB_Sig1 = (cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=1))
    #L_GB_Sig3 = (cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=3))
    #L_GB_Sig5 = (cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=5))
    #Ln_GB_Sig1 = (cv2.GaussianBlur(Ln, ksize=(0, 0), sigmaX=1))
    #Ln_GB_Sig3 = (cv2.GaussianBlur(Ln, ksize=(0, 0), sigmaX=3))
    #Ln_GB_Sig5 = (cv2.GaussianBlur(Ln, ksize=(0, 0), sigmaX=5))

    #L_GB_Sig1 = (cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=1))
    #A_GB_Sig1 = (cv2.GaussianBlur(A, ksize=(0, 0), sigmaX=1))
    #B_GB_Sig1 = (cv2.GaussianBlur(B, ksize=(0, 0), sigmaX=1))

    # LAB Difference of Gaussians
    #DoG_L2 = L - (cv2.GaussianBlur(L, ksize=(0, 0), sigmaX=6))
    #DoG_A = (cv2.GaussianBlur(A, ksize=(0, 0), sigmaX=1)) - (cv2.GaussianBlur(A, ksize=(0, 0), sigmaX=3))
    #DoG_B = (cv2.GaussianBlur(B, ksize=(0, 0), sigmaX=1)) - (cv2.GaussianBlur(B, ksize=(0, 0), sigmaX=3))




    ##### RGB (Original/Normalized)
    #img_float = img.astype(np.float32)
    #b_n, g_n, r_n = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    #total = r_n + g_n + b_n + 1e-6
    #r_n = (r_n / total)
    #g_n = (g_n / total)
    #b_n = (b_n / total)

    # RGB Spectral Indices
    #exg = (2 * g_n - r_n - b_n)
    # exr = ((1.4 * r_n) - g_n)
    # exgr = exg - exr
    #sci = ((r_n - b_n) / (r_n + b_n + 1e-6))
    # vari = (g_n-r_n)/(g_n+r_n-b_n + 1e-6)


    ### Texture Features ##############################################################################################
    ### Rugosity / Local Variance ####################################################
    #localVar_1 = cv2.GaussianBlur(L ** 2, (0, 0), sigmaX=1) - cv2.GaussianBlur(L, (0, 0), sigmaX=1) ** 2
    #localVar_5 = cv2.GaussianBlur(L ** 2, (0, 0), sigmaX=5) - cv2.GaussianBlur(L, (0, 0), sigmaX=5) ** 2
    #varRatio = localVar_1 / (localVar_5 + 1e-6)
    #localVar_11px = cv2.blur(L_CLAHE**2, (11,11))  - cv2.blur(L_CLAHE, (11,11))**2


    ### --- EDGE DETECTION --- ########################################################
    # Scharr
    #scharr_x = cv2.Scharr(L, cv2.CV_32F, 1, 0)
    #scharr_y = cv2.Scharr(L, cv2.CV_32F, 0, 1)
    #scharr_mag = np.sqrt(scharr_x ** 2 + scharr_y ** 2)
    #scharr_angle = np.arctan2(scharr_y, scharr_x).astype(np.float32)
    #scharr_A = scharr(A)
    #scharr_B = scharr(B)
    #scharr_L = scharr(L)
    #laplace_L = laplace(L)
    #laplace_Ln = cv2.normalize(laplace_L, None, 0, 255, cv2.NORM_MINMAX)
    #scharr_L_GB = cv2.GaussianBlur(scharr_L, ksize=(0,0), sigmaX=2)
    #laplace_L_GB = cv2.GaussianBlur(scharr_L, ksize=(0, 0), sigmaX=5)
    #scharr_gb = cv2.GaussianBlur(scharr_mag, ksize=(0,0), sigmaX=1)


    # Structure Tensor ####################################
    def extract_structure_features(img, sigma):
        Axx, Axy, Ayy = structure_tensor(img, sigma)
        l1, l2 = structure_tensor_eigenvalues([Axx, Axy, Ayy])
        coherence = ((l1 - l2) / (l1 + l2 + 1e-6))
        anisotropy = l1 / (l2 + 1e-6)
        return coherence, anisotropy

    TC_1, TA_1 = extract_structure_features(L.astype(np.float32), sigma=1)
    #TC_2, TA_2 = extract_structure_features(L.astype(np.float32), sigma=2)
    #TC_2, TA_2 = extract_structure_features(L.astype(np.float32), sigma=1.5)
    #coherence2, ani2 = extract_structure_features(L_CLAHE.astype(np.float32), sigma=1.5)
    #coherence3, ani3 = extract_structure_features(L_CLAHE.astype(np.float32), sigma=2)
    #coherence2= extract_structure_features(A.astype(np.float32), sigma=0.5)
    #coherence3= extract_structure_features(B.astype(np.float32), sigma=0.5)
    #l1_sig3, l2_sig3, coherence_sig3 = extract_structure_features(L_CLAHE.astype(np.float32), sigma=3)
    #l1 = np.max(np.stack([l1_sig1, l1_sig2, l1_sig3], axis=0), axis=0)
    #l2 = np.max(np.stack([l2_sig1, l2_sig2, l2_sig3], axis=0), axis=0)
    #coherence = np.max(np.stack([coherence_sig1, coherence_sig2, coherence_sig3], axis=0), axis=0)


    # Hessian Matrix #######################################
    def hess_func(img, sigma):
        hess = hessian_matrix(img, sigma=sigma, use_gaussian_derivatives=False)
        hl1, hl2 = hessian_matrix_eigvals(hess)
        hl1 = np.abs(hl1)
        hl2 = np.abs(hl2)
        coherence = ((hl1 - hl2) / (hl1 + hl2 + 1e-6))
        anisotropy = hl1 / (hl2 + 1e-6)
        return coherence, anisotropy

    #HC_05, HA_05, HR_05 = hess_func(L_CLAHE.astype(np.float32), sigma=0.5)
    HC_1, HA_1 = hess_func(L.astype(np.float32), sigma=1)
    #HC_2, HA_2 = hess_func(L.astype(np.float32), sigma=2)
    #HC_2, HA_2 = hess_func(L.astype(np.float32), sigma=1.5)
    #HC_A, HA_A = hess_func(A.astype(np.float32), sigma=1)
    #HC_B, HA_B = hess_func(B.astype(np.float32), sigma=1)
    #ridge_c = hess_func(L_CLAHE.astype(np.float32), sigma=1.5)
    #ridge_d = hess_func(L_CLAHE.astype(np.float32), sigma=2.0)
    #ridge_d, blob_d, flatness_d = hess_func(L_CLAHE.astype(np.float32), sigma=6.0)
    #ridge_e, blob_e, flatness_e = hess_func(L_CLAHE.astype(np.float32), sigma=12.0)


    # Circular Variance Equation #################################
    def extract_circvar(img, sigma):
        dx = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(img, cv2.CV_32F, 0, 1)
        angles = np.arctan2(dy, dx)
        sum_cos = cv2.GaussianBlur(np.cos(angles), (0, 0), sigmaX=sigma)
        sum_sin = cv2.GaussianBlur(np.sin(angles), (0, 0), sigmaX=sigma)
        R = np.sqrt(sum_cos ** 2 + sum_sin ** 2)
        R = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
        return R

    #CV_1= extract_circvar(L, sigma=1)
    #CV_3= extract_circvar(L, sigma=3)


    # Distance Transform (Distance to Edge) #######################
    ### Canny Edge Mapping
    #edges = cv2.Canny(L.astype(np.uint8), 25, 50)
    #edges_f = edges.astype(np.float32)
    #edge_density = cv2.GaussianBlur(scharr_L, (0, 0), 3)
    #edge_variance = cv2.blur(scharr_L ** 2, (11, 11)) - cv2.blur(scharr_L, (11, 11)) ** 2

    ### Distance Transform
    def extract_distance_transform(img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv_otsu_edges = cv2.bitwise_not(thresh)
        dist_map = cv2.distanceTransform(inv_otsu_edges, cv2.DIST_L2, 3)
        return dist_map.astype(np.float32)

    #dist = extract_distance_transform(L.astype(np.uint8))
    #distb = cv2.GaussianBlur(dist, ksize=(0, 0), sigmaX=3)


    ### Local Statistics (Mean / Std / Coefficient of Variation) #####################
    def local_stats(channel, sigma):
        ch = channel.astype(np.float32)
        local_mean = cv2.GaussianBlur(ch, (0, 0), sigma)
        local_mean2 = cv2.GaussianBlur(ch ** 2, (0, 0), sigma)
        local_var = np.maximum(local_mean2 - local_mean ** 2, 0)
        local_std = np.sqrt(local_var)
        cov = local_std / (local_mean + 1e-6)
        return local_mean, local_var, cov

    #mean_l, var_l, cov_l = local_stats(L, 3)
    #mean_A, std_A, cov_A = local_stats(A, 1)
    #mean_B, std_B, cov_B = local_stats(B, 1)

    #kernel = np.ones((5, 5), np.uint8)
    #L_max = cv2.dilate(L.astype(np.uint8), kernel).astype(np.float32)
    #L_min = cv2.erode(L.astype(np.uint8), kernel).astype(np.float32)
    #ldr = (L_max - L_min) / (L_max + L_min + 1e-6)





    # --- FINAL FEATURE STACK --- ######################################################
    return np.column_stack([
        L.ravel(),
        Ln.ravel(),
        A.ravel(),
        B.ravel(),
        TC_1.ravel(),
        HC_1.ravel(),
        L_B3.ravel(),
        L_B6.ravel(),
        L_B9.ravel()
    ]).astype(np.float32)


# Define feature (column) names ############################################################
feature_names = [
    "L (LAB)",
    "L Z-Normalized (LAB)",
    "A (LAB)",
    "B (LAB)",
    "Tensor Coherence (Sigma 1)",
    "Hessian Coherence (Sigma 1)",
    "Bilateral Blur (Sigma 3)",
    "Bilateral Blur (Sigma 6)",
    "Bilateral Blur (Sigma 9)"
]

### PROCESS IMAGERY ###################################################################################################
def process_directory(directory, SAMPLES_PER_CLASS):
    X_list, y_list = [], []
    json_files = [f for f in os.listdir(directory) if f.endswith(".json")]

    print(f"Processing {len(json_files)} images in {os.path.basename(directory)}...")

    # --- STEP 1: FILL THE LISTS FIRST ---
    for file in json_files:
        with open(os.path.join(directory, file), "r") as f:
            data = json.load(f)

        #is_burned, day = parse_metadata(data["imagePath"])
        img_file = os.path.join(directory, data["imagePath"])
        img = cv2.imread(img_file)
        if img is None: continue #days, burned, unit_id = parse_metadata(data["imagePath"])


        # Create mask
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for shape in data["shapes"]:
            label_name = shape["label"]
            points = np.array(shape["points"], dtype=np.int32)
            class_id = CLASS_MAP.get(label_name, 0)
            if shape["shape_type"] == "polygon":
                cv2.fillPoly(mask, [points], color=class_id)
            elif shape["shape_type"] in ["linestrip", "line"]:
                cv2.polylines(mask, [points], False, color=class_id, thickness=LINE_THICKNESS)

        X_list.append(extract_features(img)) # days, burned, unit_id
        y_list.append(mask.ravel())

    # --- STEP 2: STACK AFTER THE LOOP ---
    if not X_list:
        raise ValueError(f"No valid data found in {directory}.")

    X_stacked = np.vstack(X_list)
    y_stacked = np.concatenate(y_list)


    # --- STEP 3: SAMPLE ---
    final_X, final_y = [], []

    for class_id in range(1, 7):  # Ignore Background Class (0)
        indices = np.where(y_stacked == class_id)[0]
        if len(indices) == 0: continue

        # USE TO DEFINE SPLIT-VEGETATION PERCENTAGES (Currently doing natural selection on pre-merged vegetation)
        #if class_id == 1:  # Green Vegetation
            # Take 70% of the Veg budget
            #n = min(len(indices), int(SAMPLES_PER_CLASS * 0.6))
        #elif class_id == 6:  # Senesced Vegetation
            # Take 30% of the Veg budget
            #n = min(len(indices), int(SAMPLES_PER_CLASS * 0.4))
        #else:
            #n = min(len(indices), SAMPLES_PER_CLASS)

        n = min(len(indices), SAMPLES_PER_CLASS)

        selected_indices = np.random.choice(indices, n, replace=False)
        final_X.append(X_stacked[selected_indices])
        final_y.append(y_stacked[selected_indices])

    # --- STEP 4: CONSOLIDATE AND MERGE ---
    X_final = np.vstack(final_X)
    y_final = np.concatenate(final_y)

    #y_final[y_final == 6] = 1  # Merge early and late-season vegetation (only if using defined percentages)

    return X_final, y_final



### Train Model ######################################################################################################

### MODEL FOR IMAGE LEVEL 80/20 SPLIT. USE PIXEL LEVEL SPLIT (BELOW)
'''
# 3. Train Model
print(f"Training on {len(y_train)} pixels...")
model = RandomForestClassifier(n_estimators=150,
                               n_jobs=-1,
                               #class_weight={1:1, 2:1, 3:1, 4:1, 5:1},
                               #max_features='log2', # Force it to look at a smaller subset of features for each split
                               min_samples_leaf=5,
                               max_depth=20,
                               random_state=99).fit(X_train, y_train)

# 4. Save model
joblib.dump(model, 'MSR_QuadPhoto_PixClassifier.pkl')


# Upload saved model to avoid retraining
#model = joblib.load('MSR_QuadPhoto_PixClassifier.pkl')

# 5. Validation Report
### Predict on scaled validation data
y_pred_val = model.predict(X_val)

print("\n--- Model Accuracy Report (Validation) ---")
print(f"Tested on {len(y_val)} pixels from unseen images in Validation folder.")

# Map target names correctly based on classes present in validation set
presentClasses = np.unique(y_val)
target_names = [CLASS_NAMES[i] for i in presentClasses]
print(classification_report(y_val, y_pred_val, target_names=target_names))

# 6. Variable Importance
# Calculate & sort feature importances
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the results
print("\n--- Feature Importance Ranking ---")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

# Print model performance (R2)
r2 = r2_score(y_val, y_pred_val)
print(f"\nTest R-squared score: {r2:.3f}")
'''


### PIXEL-LEVEL 80/20 SPLIT WITH 5-FOLD VALIDATION #####################################################################
X_train, y_train = process_directory(TEST_DIR, SAMPLES_PER_CLASS=100000) # Define the number of training pixels per class

# --- 3. CALL THE CROSS-VALIDATION HERE ---
print("Starting Thesis Validation...")
final_model = run_random_5fold(X_train, y_train)

importances = final_model.feature_importances_
indices = np.argsort(importances)[::-1]
# Print the results
print("\n--- Feature Importance Ranking ---")
for i, idx in enumerate(indices):
    print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")



### Batch Predict Photos ###############################################################################################
'''
stats = []
print(f"Processing images in {AP_INPUT}...")

# 1. Get the list of images first so tqdm knows the total count
image_list = glob.glob(os.path.join(AP_INPUT, "*.jpg"))

# 2. Wrap the list in tqdm for the progress bar
for img_path in tqdm(image_list, desc="Batch Predicting", unit="img"):
    img = cv2.imread(img_path)
    if img is None: continue
    
    # Feature Extraction
    filename = os.path.basename(img_path)
    #burned, day = parse_metadata(filename)
    X_test = extract_features(img)


    # Predict full image and reshape to 2D
    raw_preds = final_model.predict(X_test).reshape(img.shape[:2]).astype(np.uint8)
    # 3. THE CLEAN: 3x3 Median Filter
    # Removes "salt and pepper" noise before stats are calculated
    preds = cv2.medianBlur(raw_preds, 3)



    # 4. Calculate percentages for each class (IDs 1 through 5)
    img_stats = {"Image": os.path.basename(img_path)}
    total_px = preds.size
    
    # We use range(1, len(CLASS_NAMES)) to ensure we only get
    # Vegetation, Soil, Litter, Substrate, and Quadrat
    for i in range(1, len(CLASS_NAMES)):
        name = CLASS_NAMES[i]
        count = np.sum(preds == i)
        img_stats[f"{name} Percent"] = round((count / total_px) * 100, 2)

    stats.append(img_stats)

    # 5. Save visual mask (mapping IDs to BGR colors)
    vis_mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    for class_id, bgr_color in CLASS_COLORS.items():
        vis_mask[preds == class_id] = bgr_color

    # Save the cleaned color map
    cv2.imwrite(os.path.join(AP_OUTPUT, os.path.basename(img_path)), vis_mask)

### Export Prediction CSV ####################
df = pd.DataFrame(stats)
df.to_csv(os.path.join(AP_OUTPUT, "MSR_MultiClass_Predictions.csv"), index=False)

print(f"\n Processed {len(stats)} images. Results saved to {AP_OUTPUT}")
'''