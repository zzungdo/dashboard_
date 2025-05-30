import os, base64, random
import pandas as pd
import numpy as np
import cv2
import plotly.express as px

FIXED_CLASS_COLORS = {
    'car': (0, 255, 0),
    'bus': (255, 127, 0),
    'truck': (255, 255, 0),
    'person': (255, 0, 0),
    'motorcycle': (255, 0, 255)
}
# *랜덤 색상 저장할 딕셔너리
class_color_map = {}

# *랜덤 색상 지정 함수
def get_random_color():
    return (
        random.randint(50, 255),
        random.randint(50, 255),
        random.randint(50, 255)
    )

# *클래스명이랑 색상 매칭 함수
def get_class_color(classname):
    if classname in FIXED_CLASS_COLORS:
        return FIXED_CLASS_COLORS[classname]
    if classname not in class_color_map:
        class_color_map[classname] = get_random_color()
    return class_color_map[classname]

# *썸네일 이미지 생성 함수
def get_image_thumbnail(filename, img_folder, thumb_size=(120, 90)):
    image_path = os.path.join(img_folder, filename)
    img = imread_unicode(image_path)

    if img is None:
        return ''
    h, w = img.shape[:2]
    scale = min(thumb_size[0]/w, thumb_size[1]/h)
    new_w, new_h = int(w*scale), int(h*scale)
    thumb = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    _, buffer = cv2.imencode('.jpg', thumb)
    # *웹에서 이미지 띄우려고 변환
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

# *이상 경로에 대비한 함수
def imread_unicode(path):
    # * 경로에 한글,특수문자 있는지 판단 함수
    def is_unicode_path(p):
        try:
            p.encode('ascii')
            return False
        except UnicodeEncodeError:
            return True
    # *일반 경로일때
    if not is_unicode_path(path):
        return cv2.imread(path)

    try:
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[읽기 실패] {path}")
        return None

# *GT 로딩 및 정규화
def load_all_gt(gt_folder, img_folder):
    all_data = []

    for file in os.listdir(gt_folder):
        if file.endswith('.txt'):
            label_path = os.path.join(gt_folder, file)
            image_name = os.path.splitext(file)[0] + '.jpg'
            image_path = os.path.join(img_folder, image_name)
            # *이미지 없으면 넘어감
            if not os.path.exists(image_path):
                continue
            # *이미지 로딩 실패하면 넘어감
            image = imread_unicode(image_path)
            if image is None:
                continue

            h, w = image.shape[:2]

            df = pd.read_csv(label_path, header=None, names=['class', 'xmin', 'ymin', 'xmax', 'ymax'])

            df['filename'] = image_name
            df['img_width'] = w
            df['img_height'] = h
            df['center_x'] = (df['xmin'] + df['xmax']) / 2
            df['center_y'] = (df['ymin'] + df['ymax']) / 2
            df['area'] = (df['xmax'] - df['xmin']) * (df['ymax'] - df['ymin'])
            df['ratio'] = (df['xmax'] - df['xmin']) / (df['ymax'] - df['ymin'] + 1e-6)

            # *정규화 좌표
            df['center_x_norm'] = df['center_x'] / w
            df['center_y_norm'] = df['center_y'] / h
            df['xmin_norm'] = df['xmin'] / w
            df['xmax_norm'] = df['xmax'] / w
            df['ymin_norm'] = df['ymin'] / h
            df['ymax_norm'] = df['ymax'] / h
            df['area_norm'] = df['area'] / (w * h)

            df['bbox_width'] = df['xmax'] - df['xmin']
            df['bbox_height'] = df['ymax'] - df['ymin']

            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# *이미지별 요약 생성 함수
def compute_image_summary(df):
    return df.groupby('filename').agg(
        bbox_count=('class', 'count'), # *bbox 총 개수
        unique_classes=('class', pd.Series.nunique), # *클래스 개수
        included_classes=('class', lambda x: ', '.join(sorted(set(x))))
    ).reset_index()

# *히트맵 관련 함수
def compute_heatmap_data(df, grid_size=5):

    df = df.dropna(subset=['xmin_norm', 'xmax_norm', 'ymin_norm', 'ymax_norm'])

    heatmap = np.zeros((grid_size, grid_size), dtype=int)

    for _, row in df.iterrows():
        # *좌표 → 그리드 인덱스로 변환 (경계 포함 + 안전 처리)
        xmin_i = int(np.floor(row['xmin_norm'] * grid_size))
        xmax_i = int(np.floor(row['xmax_norm'] * grid_size))
        ymin_i = int(np.floor(row['ymin_norm'] * grid_size))
        ymax_i = int(np.floor(row['ymax_norm'] * grid_size))

        # *경계 보정 (grid index 범위: 0 ~ grid_size - 1)
        xmin_i = max(0, min(xmin_i, grid_size - 1))
        xmax_i = max(0, min(xmax_i, grid_size - 1))
        ymin_i = max(0, min(ymin_i, grid_size - 1))
        ymax_i = max(0, min(ymax_i, grid_size - 1))

        # *bbox가 덮는 모든 셀에 +1
        for i in range(xmin_i, xmax_i + 1):
            for j in range(ymin_i, ymax_i + 1):
                heatmap[i, j] += 1

    # *시각화를 위한 DataFrame 변환
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    df_heat = pd.DataFrame({
        'grid_x': x.flatten(),
        'grid_y': y.flatten(),
        'box_count': heatmap.T.flatten()
    })

    return df_heat[df_heat['box_count'] > 0].copy()


# *바운딩박스 시각화

def draw_boxes(filename, gt_folder, img_folder, boxes, visible_classes=None, duplicate=False):
    # 1) 이미지 경로 생성
    image_path = os.path.join(img_folder, filename)
    # 2) 이미지 로드 (한글/특수문자 경로 지원)
    image = imread_unicode(image_path)
    if image is None:
        return None

    # 3) 바운딩박스 그리기
    for idx, row in boxes.iterrows():
        if visible_classes and row['class'] not in visible_classes:
            continue

        # 중복 검사 모드라면 색상 토글
        if duplicate:
            color = [(255,0,0),(0,255,0)][idx % 2]
        else:
            color = get_class_color(row['class'])

        pt1 = (int(row['xmin']), int(row['ymin']))
        pt2 = (int(row['xmax']), int(row['ymax']))
        label = row['class']
        w = int(row['xmax'] - row['xmin'])
        h = int(row['ymax'] - row['ymin'])

        cv2.rectangle(image, pt1, pt2, color, 2)
        cv2.putText(image, f"{label} ({w}x{h})",
                    (pt1[0], pt1[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 4) Base64 인코딩
    _, buffer = cv2.imencode('.jpg', image)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"