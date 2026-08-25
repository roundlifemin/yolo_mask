
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from ultralytics import YOLO
import av
import numpy as np
import cv2
import tempfile
import logging

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
    return YOLO("best.pt")  # 코드 파일과 같은 경로에 best.pt가 존재해야 함

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
    return results[0].plot()  # 추론 결과 시각화(Bounding Box 추가) 후 반환

# -------------------------------------------------------------------
# 4. 사이드바 탐지 모드 선택 (이미지 / 웹캠 / 동영상)
# -------------------------------------------------------------------
mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠", "동영상"])

# ===================================================================
# MODE 1: 이미지 업로드 탐지
# ===================================================================
if mode == "이미지":
    # 사용자로부터 이미지 파일 입력받기
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        # 업로드된 바이트 데이터를 OpenCV 이미지(BGR) 형식으로 변환
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # 원본 이미지 출력 (OpenCV BGR -> Streamlit RGB 변환 필요)
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_container_width=True)

        # 탐지 실행 및 결과 화면 출력
        st.subheader("탐지 결과")
        result_bgr = detect_image(image_bgr)
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), caption="탐지된 이미지", use_container_width=True)

# ===================================================================
# MODE 2: 실시간 웹캠 스트리밍 탐지 (WebRTC 활용)
# ===================================================================
elif mode == "웹캠":
    # WebRTC 비디오 프레임 처리 클래스 정의
    class VideoTransformer(VideoTransformerBase):
        def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
            # 입력된 비디오 프레임을 NumPy 배열(BGR)로 변환
            img = frame.to_ndarray(format="bgr24")
            
            # YOLOv8 추론 실행
            result = detect_image(img)
            
            # 추론 결과를 다시 VideoFrame 객체로 변환하여 리턴
            return av.VideoFrame.from_ndarray(result, format="bgr24")

    try:
        # WebRTC 스트리머 연결 설정
        webrtc_streamer(
            key="mask-detect",
            video_processor_factory=VideoTransformer,
            media_stream_constraints={"video": True, "audio": False},  # 비디오 사용, 오디오 비활성화
            rtc_configuration={
                "iceServers": [
                    # STUN 및 TURN 서버 설정 (클라우드 환경 연결 호환성 개선 목적)
                    {"urls": "stun:stun.l.google.com:19302"},
                    {
                        "urls": "turn:openrelay.metered.ca:80",
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    },
                ]
            },
            async_processing=True,  # 비동기 프레임 처리 적용
        )
    except Exception as e:
        st.error(f"웹캠 스트리밍 실행 중 오류 발생: {e}")
        st.info("Streamlit Cloud 환경에서는 TURN/STUN 연결 문제로 웹캠 스트리밍이 실패할 수 있습니다. 이미지 업로드 모드를 권장합니다.")

# ===================================================================
# MODE 3: 동영상 파일 탐지 및 재생
# ===================================================================
elif mode == "동영상":
    # 사용자로부터 비디오 파일 입력받기
    uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi"])
    
    if uploaded_video:
        # OpenCV VideoCapture 활용을 위해 임시 파일(Temp File)로 저장
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        # 실시간 연속 렌더링을 위한 빈 플레이스홀더 생성
        stframe = st.empty()
        st.subheader("탐지 결과 (실시간 재생)")

        # 비디오 프레임을 순차적으로 읽어와 탐지 결과 업데이트
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 비디오 재생이 종료되면 루프 탈출
                
            # 프레임별 마스크 탐지
            result_bgr = detect_image(frame)
            
            # BGR -> RGB 변환 후 화면 갱신
            stframe.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # 사용이 끝난 비디오 캐처 리소스 해제
        cap.release()
