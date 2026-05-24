import os, cv2, numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops,local_binary_pattern
import joblib


#详细的特征提取方法
def glcm_features(gray_img, distances=[1,3,5], angles=[0, np.pi/4, np.pi/2]):
    glcm = graycomatrix(gray_img, distances=distances, angles=angles,
                        levels=256, symmetric=True, normed=True)
    features = []
    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    for prop in props:
        feat = graycoprops(glcm, prop)
        features.extend(feat.flatten())
    return np.array(features)



def lbp_hist(gray_img, P=8, R=1, eps=1e-7):
    lbp = local_binary_pattern(gray_img, P, R, method="uniform")
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist

def hu_moments(img):
    # img 可以是灰度图或二值图，先计算力矩
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    # 为避免零的极端小值，取对数绝对值
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu

def color_stats(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mean_rgb = img_bgr.mean(axis=(0,1))
    std_rgb = img_bgr.std(axis=(0,1))
    mean_hsv = hsv.mean(axis=(0,1))
    std_hsv = hsv.std(axis=(0,1))
    return np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv])

# 特征提取汇总
def extract_features(img_path, size=(256, 256)):
    img = cv2.imread(img_path)
    if img is None: return None
    img = cv2.resize(img, size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 纹理
    f_glcm = glcm_features(gray)
    f_lbp = lbp_hist(gray)
    # 形状 (Hu矩对二值图)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    f_hu = hu_moments(binary)
    # 颜色
    f_color = color_stats(img)

    return np.concatenate([f_glcm, f_lbp, f_hu, f_color])


# 加载数据集
base_path = "Trash_floaters_exist_classify"
X, y = [], []
for label, class_name in enumerate(["no_floaters", "have_floaters"]):
    folder = os.path.join(base_path, class_name)
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        feats = extract_features(path)
        if feats is not None:
            X.append(feats)
            y.append(label)

X = np.array(X)
y = np.array(y)

# 划分训练/测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM 调参
svc = SVC(kernel='rbf', probability=True, random_state=42)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}
grid = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)
print("最佳参数:", grid.best_params_)
print("最佳交叉验证准确率: {:.4f}".format(grid.best_score_))


# 测试与可靠性评估
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n测试集分类报告:")
print(classification_report(y_test, y_pred, target_names=["no_floaters", "have_floaters"]))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", cm)


# 保存模型、标准化器以及必要参数
model_assets = {
    'svm': best_model,          # 最佳SVM模型
    'scaler': scaler,           # 标准化器
    'image_size': (256, 256),   # 与训练时一致的尺寸
}
joblib.dump(model_assets, 'floaters_svm_model.joblib')
print("模型已保存为 floaters_svm_model.joblib")



# Fitting 5 folds for each of 24 candidates, totalling 120 fits
# 最佳参数: {'C': 100, 'gamma': 0.001}
# 最佳交叉验证准确率: 0.9866
#
# 测试集分类报告:
#                precision    recall  f1-score   support
#
#   no_floaters       0.99      0.97      0.98        95
# have_floaters       0.98      0.99      0.99       193
#
#      accuracy                           0.99       288
#     macro avg       0.99      0.98      0.98       288
#  weighted avg       0.99      0.99      0.99       288
#
# 混淆矩阵:
#  [[ 92   3]
#  [  1 192]]
# 模型已保存为 floaters_svm_model.joblib


