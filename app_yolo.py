import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"  # YOLO 임시 설정 경로

import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import av

# ✅ WebRTC 설정 (STUN만 사용 / 로컬 환경에선 충분)
from streamlit_webrtc import RTCConfiguration
rtc_config = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ✅ Streamlit 설정
st.set_page_config(page_title="YOLOv8 마스크 탐지", layout="centered")
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

# ---------------------------
# 모델 불러오기 (캐시)
# ---------------------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # same dir에 best.pt 있어야 함

model = load_model()

# ---------------------------
# 사이드바 옵션
# ---------------------------
st.sidebar.header("옵션")
conf = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
imgsz = st.sidebar.selectbox("이미지 크기(imgsz)", [320, 416, 512, 640, 800], index=3)
mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠(브라우저)", "동영상"])

# ---------------------------
# 이미지 추론 함수
# ---------------------------
def detect_image(image_rgb, conf=0.25, imgsz=640):
    results = model(image_rgb, conf=conf, imgsz=imgsz)
    vis_bgr = results[0].plot()
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return vis_rgb

# ---------------------------
# 웹캠용 YOLO 클래스
# ---------------------------
class YoloTransformer(VideoTransformerBase):
    def __init__(self, conf=0.25, imgsz=640):
        self.conf = conf
        self.imgsz = imgsz
        self.model = model

    def transform(self, frame: av.VideoFrame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.model(img_rgb, conf=self.conf, imgsz=self.imgsz)
        vis_bgr = results[0].plot()
        return vis_bgr

# ---------------------------
# 1) 이미지 모드
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
# 2) 웹캠(브라우저) 모드
# ---------------------------
elif mode == "웹캠(브라우저)":
    st.info("브라우저 카메라를 사용합니다. 접근 권한을 허용해주세요. (로컬 실행 시 정상 작동)")
    webrtc_streamer(
        key="yolo-webrtc",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_transformer_factory=lambda: YoloTransformer(conf=conf, imgsz=imgsz),
    )

# ---------------------------
# 3) 동영상 모드
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
