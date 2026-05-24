import os, cv2, numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

# ===== 配置 =====
IMAGE_SIZE = (200, 100)      # 将大小不一的原图统一缩放至该尺寸（像素维度：100×100=10000）
C = 100                      # 已知最佳参数
gamma = 0.001


# 加载数据集：灰度图 → 缩放 → 展平（直接用像素作为特征）
base_path = "Trash_floaters_exist_classify_enhance_cut"
X, y = [], []
for label, class_name in enumerate(["no_floaters", "have_floaters"]):
    folder = os.path.join(base_path, class_name)
    for fname in os.listdir(folder):
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, IMAGE_SIZE)
        X.append(img.flatten())
        y.append(label)

X = np.array(X, dtype=np.float64)
y = np.array(y)

# 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用固定参数的 SVM 模型
svm = SVC(kernel='rbf', C=C, gamma=gamma, probability=True, random_state=42)

# 交叉验证（5折，与原始代码保持一致）
cv_scores = cross_val_score(svm, X_train, y_train, cv=5, scoring='accuracy')
print(f"使用参数: {{'C': {C}, 'gamma': {gamma}}}")
print(f"交叉验证准确率: {cv_scores.mean():.4f}")

# 在完整训练集上训练最终模型
svm.fit(X_train, y_train)

# 测试集评估
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)[:, 1]

print("\n测试集分类报告:")
print(classification_report(y_test, y_pred, target_names=["no_floaters", "have_floaters"]))

cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", cm)

