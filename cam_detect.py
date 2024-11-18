import cv2
import numpy as np
import sounddevice as sd
import random
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO("")

# 웹캠 사용
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 클래스 ID를 주파수 범위(Hz)와 매핑
frequency_range_map = {
    'magpie': (2000, 5000),      # 까치가 감지되었을 때 2000~5000Hz 소리
    'waterdeer': (50, 100),      # 고라니가 감지되었을 때 50~200Hz 초저음
    'wildboar': (20, 40)         # 멧돼지가 감지되었을 때 20~40Hz 초저음
}

# 주파수 범위 내에서 무작위 톤을 재생하는 함수
def play_random_tone(frequency_range, duration=0.5, sample_rate=44100):
    frequency = random.uniform(*frequency_range)  # 주파수 범위 내 무작위 선택
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)  # 사인파 생성
    sd.play(wave, sample_rate)  # 소리 재생
    sd.wait()  # 소리 재생이 끝날 때까지 대기

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # YOLO 모델을 사용한 객체 감지
    results = model(frame)

    # 새로운 이미지를 생성하고, 신뢰도 >= 0.6인 객체만 그리기
    filtered_frame = frame.copy()  # 원본 프레임 복사
    for result in results:
        for obj in result.boxes:
            confidence = obj.conf.item()  # 객체 신뢰도를 float로 변환
            if confidence >= 0.6:
                class_id = obj.cls
                class_name = model.names[int(class_id)]  # 클래스 이름 가져오기
                print(f"감지됨: {class_name}, 신뢰도: {confidence:.2f}")

                # 감지된 객체 클래스에 따른 주파수 범위 내 무작위 톤 재생
                frequency_range = frequency_range_map.get(class_name)
                if frequency_range:
                    play_random_tone(frequency_range)

                # 신뢰도 >= 0.6인 객체에만 경계 상자 및 레이블 그리기
                box = obj.xyxy[0].cpu().numpy().astype(int)  # 경계 상자 좌표
                cv2.rectangle(filtered_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.putText(filtered_frame, f"{class_name} ({confidence:.2f})", 
                            (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

    # 신뢰도 >= 0.6인 객체만 표시
    cv2.imshow("YOLO 실시간 감지", filtered_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
