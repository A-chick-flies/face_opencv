import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from plyer import notification
import torch
import pickle
from scipy.spatial.distance import cosine

# MTCNN 얼굴 탐지기
mtcnn = MTCNN(keep_all=True)

# Facenet 모델
model = InceptionResnetV1(pretrained='vggface2').eval()

# 얼굴 임베딩을 저장한 파일 경로
embedding_file = "face_embeddings.pkl"

# 저장된 얼굴 임베딩 불러오기
with open(embedding_file, "rb") as f:
    registered_embeddings = pickle.load(f)

# 웹캡처
cap = cv2.VideoCapture(0)

# 알림을 이미 보냈는지 추적하는 변수
notified_faces = {}

while True:
    ret, frame = cap.read()

    # 얼굴 탐지
    result = mtcnn.detect(frame)
    
    if len(result) == 2:
        boxes, probs = result
    else:
        boxes, probs, landmarks = result

    if boxes is not None:
        for box in boxes:
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        # 얼굴 임베딩 추출
        faces = mtcnn(frame)
        if faces is not None:
            for face in faces:
                # Facenet 모델을 통해 얼굴 특징 벡터 생성
                embedding = model(face.unsqueeze(0))

                # 얼굴 유사도 계산 (cosine similarity)
                embedding = embedding.detach().numpy().flatten()
                for name, registered_embedding in registered_embeddings.items():
                    similarity = cosine(registered_embedding, embedding)

                    # 유사도가 일정 기준 이하일 경우 (예: 0.6 이하)
                    if similarity < 0.6:
                        # 이미 알림을 보낸 적이 없다면 알림을 보냄
                        if name not in notified_faces:
                            notification.notify(
                                title=f"{name} 인식됨!",
                                message=f"{name}가 인식되었습니다.",
                                timeout=5  # 알림이 5초 동안 표시됨
                            )
                            notified_faces[name] = True  # 알림을 보냈다고 기록
                            print(f"{name} 인식됨!")

    # 프레임 출력
    cv2.imshow('Face Recognition', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 캡처 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
