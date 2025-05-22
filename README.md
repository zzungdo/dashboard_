# Bounding Box 분석 및 시각화 대시보드

Python 기반 Dash 웹 앱으로, 바운딩박스(GT 파일) 데이터를 시각화하고  
중복 여부, 클래스 분포, 위치 통계 등을 확인할 수 있는 도구입니다.

---

## 실행 순서 요약

압축을 풀고, 아래 순서대로 실행하면 됩니다:

1. Python 3.9 가상환경 만들기:
   ```bash
   conda create -n dashboard_env python=3.9 -y && conda activate dashboard_env
   ```

2. 패키지 설치:
   ```bash
   pip install -r requirements.txt
   ```

3. 앱 실행:
   ```bash
   python app.py
   ```

---

## 데이터셋 경로 설정

- `utils.py` 상단에서 GT 및 이미지 경로를 직접 지정합니다.
- NAS 또는 공유 폴더 사용 시, 반드시 해당 경로를 **Windows 탐색기에서 로컬 드라이브(Z:\\ 등)** 으로 마운트해서 사용해주세요.  
  (UNC 경로 `\\192.168...` 형태는 오류가 발생할 수 있습니다.)

---

## 웹 브라우저 접속

앱 실행 후, 다음 중 하나의 주소로 접속합니다:

- http://localhost:8080  
- http://<실행PC의 IP>:8080 (예: http://192.168.102.XXX:8080)

---

## 사용 방법

- 탭별 주요 기능:
  - **산점도**: Bounding Box의 Width,Height를 기준으로 시각화  
  - **히트맵**: 그리드 기반 밀도 시각화  
  - **클래스 분포 / 통계**: 전체 클래스 비율 및 개수 시각화  
  - **이미지 요약 / 확인**: Image 내 정보 요약 및 Viewing  
  - **데이터 검사**: 클래스/좌표 기준 중복 바운딩박스 탐지

- Image Viewer
  - **Zoom In** : 마우스 좌클릭 → 드래그하여 사각형 형태로 확대 (box zoom)  
  - **Zoom Out**: 더블클릭 → 원래 크기로 복귀  
  - **툴바** :  Zoom, Pan 등 설정  
    - 마우스 휠 줌은 **비활성화됨** (좌우 잘림 방지 목적)  
  - 산점도, 히트맵, 테이블 클릭 시 하단 이미지 자동 전환

---

## 패키지 설치

이 프로젝트에 필요한 모든 패키지는 `requirements.txt`에 포함되어 있습니다.  
아래 명령어로 한 번에 설치할 수 있습니다:

```bash
pip install -r requirements.txt
```

---

## 개발 환경 정보

- Python: 3.13.2 (Anaconda)  
- 가상환경: `dashboard_env`  
- OS: Windows 10 64bit

> 본 프로젝트는 Python 3.13.2에서 개발되었으며, 최신 Python 환경에서 동작합니다.  
> 단, 일부 PC에서는 Python 3.9 또는 3.10 버전이 더 안정적으로 작동할 수 있으므로 해당 버전 사용도 가능합니다.

---
## Tree 구조
```
dashboard_v1.0.0
 ┣ app.py
 ┣ callbacks.py
 ┣ layout.py
 ┣ README.md
 ┣ requirements.txt
 ┗ utils.py
 ```