import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"  # 쓰기 가능한 경로

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# 브라우저 웹캠용
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av  # streamlit-webrtc가 사용하는 영상 프레임 타입

st.set_page_config(page_title="YOLOv8 마스크 탐지", layout="centered")
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

# ---------------------------
# 모델 불러오기 (캐시)
# ---------------------------
@st.cache_resource
def load_model():
    # 같은 디렉토리에 best.pt 필요
    return YOLO("best.pt")

model = load_model()

# ---------------------------
# 공통 옵션 (사이드바)
# ---------------------------
st.sidebar.header("옵션")
conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.selectbox("이미지 크기(imgsz)", [320, 416, 512, 640, 800], index=3)

# ---------------------------
# 이미지 추론 함수
# (RGB 입력 -> Annotated RGB 반환)
# ---------------------------
def detect_image(image_rgb, conf=0.25, imgsz=640):
    results = model(image_rgb, conf=conf, imgsz=imgsz)   # YOLOv8 예측
    vis_bgr = results[0].plot()                          # BGR(주석 포함)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)   # 스트림릿 표시에 맞게 RGB로
    return vis_rgb

# ---------------------------
# 브라우저 웹캠용 Transformer
# ---------------------------
class YoloTransformer(VideoTransformerBase):
    def __init__(self, conf=0.25, imgsz=640):
        self.conf = conf
        self.imgsz = imgsz
        self.model = model  # 캐시된 모델 재사용

    def transform(self, frame: av.VideoFrame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # YOLO 추론
        results = self.model(img_rgb, conf=self.conf, imgsz=self.imgsz)
        vis_bgr = results[0].plot()  # BGR로 주석 렌더링

        # streamlit-webrtc는 BGR ndarray 반환 가능
        return vis_bgr

# ---------------------------
# 탐지 모드 선택
# ---------------------------
mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠(브라우저)", "동영상"])

# ---------------------------
# 1) 이미지
# ---------------------------
if mode == "이미지":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        st.image(image_rgb, caption="원본 이미지", use_container_width=True)
        st.subheader("탐지 결과")

        result_rgb = detect_image(image_rgb, conf=conf, imgsz=imgsz)
        st.image(result_rgb, caption="탐지된 이미지", use_container_width=True)

# ---------------------------
# 2) 웹캠(브라우저)
# ---------------------------
elif mode == "웹캠(브라우저)":
    st.info("브라우저 카메라를 사용합니다. 접근 권한을 허용해주세요.")
    # 현재 슬라이더 값을 transformer에 전달하기 위해 factory로 주입
    webrtc_streamer(
        key="yolo-webrtc",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=lambda: YoloTransformer(conf=conf, imgsz=imgsz),
    )

# ---------------------------
# 3) 동영상
# ---------------------------
elif mode == "동영상":
    uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        st.subheader("탐지 결과 (실시간 재생)")

        # 간단한 진행 바
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = st.progress(0)
        cur = 0

        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result_rgb = detect_image(frame_rgb, conf=conf, imgsz=imgsz)
            stframe.image(result_rgb, channels="RGB", use_container_width=True)

            cur += 1
            if total_frames > 0:
                pbar.progress(min(cur / total_frames, 1.0))

        cap.release()
        st.success("처리가 완료되었습니다.")
