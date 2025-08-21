import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import numpy as np
import cv2
import tempfile
import logging

st.set_page_config(page_title="YOLOv8 마스크 탐지", layout="centered")
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

@st.cache_resource
def load_model():
    return YOLO("best.pt")  # 반드시 같은 폴더에 best.pt 포함

model = load_model()

def detect_image(image_bgr):
    results = model(image_bgr)
    return results[0].plot()

mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠", "동영상"])

# 이미지 탐지
if mode == "이미지":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_container_width=True)

        st.subheader("탐지 결과")
        result_bgr = detect_image(image_bgr)
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), caption="탐지된 이미지", use_container_width=True)

# 웹캠 탐지
elif mode == "웹캠":
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            result = detect_image(img)
            return av.VideoFrame.from_ndarray(result, format="bgr24")

    try:
        webrtc_streamer(
            key="mask-detect",
            video_processor_factory=VideoTransformer,  # 최신 권장 방식
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
        logging.exception(e)
        st.info("Streamlit Cloud 환경에서는 TURN/STUN 연결 문제로 웹캠 스트리밍이 실패할 수 있습니다. 이미지 업로드 모드를 권장합니다.")

# 동영상 탐지
elif mode == "동영상":
    uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.subheader("탐지 결과 (실시간 재생)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_bgr = detect_image(frame)
            stframe.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            # 잠시 대기 - 너무 빠른 루프 방지
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

