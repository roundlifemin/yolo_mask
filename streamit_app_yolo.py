import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------------------------------------------------
# 1. 페이지 기본 설정 및 타이틀 출력
# -------------------------------------------------------------------
st.set_page_config(page_title="YOLOv8 마스크 탐지", layout="centered")
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

# -------------------------------------------------------------------
# 2. 모델 로드 (캐싱을 통해 중복 로딩 방지)
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    # 저장된 학습 모델(best.pt)을 로드합니다.
    return YOLO("best.pt")

model = load_model()

# -------------------------------------------------------------------
# 3. 마스크 탐지 공통 추론 함수
# -------------------------------------------------------------------
def detect_image(image_bgr):
    """
    입력받은 BGR 이미지 프레임에 대해 YOLOv8 모델 추론을 진행하고,
    바운딩 박스가 그려진 이미지 결과(BGR)를 반환합니다.
    """
    results = model(image_bgr)
    return results[0].plot()

# -------------------------------------------------------------------
# 4. 사이드바 탐지 모드 선택
# -------------------------------------------------------------------
mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠", "동영상"])

# ===================================================================
# MODE 1: 이미지 업로드 탐지
# ===================================================================
if mode == "이미지":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_container_width=True)

        st.subheader("탐지 결과")
        result_bgr = detect_image(image_bgr)
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), caption="탐지된 이미지", use_container_width=True)

# ===================================================================
# MODE 2: 실시간 웹캠 스트리밍 탐지 (WebRTC 최신 규격 반영)
# ===================================================================
elif mode == "웹캠":
    class YOLOVideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            result = detect_image(img)
            return av.VideoFrame.from_ndarray(result, format="bgr24")

    try:
        webrtc_streamer(
            key="mask-detect",
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [
                    {"urls": "stun:stun.l.google.com:19302"},
                    {
                        "urls": "turn:openrelay.metered.ca:80",
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    },
                ]
            },
            async_processing=True,
        )
    except Exception as e:
        st.error(f"웹캠 스트리밍 실행 중 오류 발생: {e}")
        st.info("Streamlit Cloud 환경에서는 TURN/STUN 연결 문제로 웹캠 스트리밍이 실패할 수 있습니다. 이미지 업로드 모드를 권장합니다.")

# ===================================================================
# MODE 3: 동영상 파일 탐지 (임시파일 안전 자동 삭제 로직 적용)
# ===================================================================
elif mode == "동영상":
    uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        # NamedTemporaryFile을 안전하게 핸들링
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            temp_path = tfile.name

        cap = cv2.VideoCapture(temp_path)
        stframe = st.empty()
        st.subheader("탐지 결과 (실시간 재생)")

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                result_bgr = detect_image(frame)
                stframe.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        finally:
            cap.release()
            # 사용 후 임시 파일 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
