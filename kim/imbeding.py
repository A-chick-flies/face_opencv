import cv2
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
import pickle

# MTCNN 얼굴 탐지기
mtcnn = MTCNN(keep_all=True)

# Facenet 모델
model = InceptionResnetV1(pretrained='vggface2').eval()

# 얼굴 임베딩을 저장할 파일 경로
embedding_file = "face_embeddings.pkl"

# 이미 등록된 얼굴 임베딩이 있으면 불러오기
if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        registered_embeddings = pickle.load(f)
else:
    registered_embeddings = {}

# 웹캡처
cap = cv2.VideoCapture(0)

# 얼굴이 등록되었는지 여부를 추적하는 변수
face_registered = False

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
        if faces is not None and not face_registered:  # 얼굴 등록이 아직 안 된 경우에만 실행
            for face in faces:
                # Facenet 모델을 통해 얼굴 특징 벡터 생성
                embedding = model(face.unsqueeze(0))

                # 얼굴 임베딩을 numpy array로 변환
                embedding = embedding.detach().numpy().flatten()

                # 이름을 입력받아 얼굴 임베딩 등록
                name = input("이 사람의 이름을 입력하세요 (입력 후 Enter 키): ")

                # 등록된 얼굴 임베딩 딕셔너리에 저장
                registered_embeddings[name] = embedding

                # 얼굴 임베딩 파일에 저장
                with open(embedding_file, "wb") as f:
                    pickle.dump(registered_embeddings, f)

                face_registered = True  # 얼굴 등록 완료 플래그 설정

                print(f"{name}의 얼굴이 등록되었습니다!")

                break  # 얼굴이 등록되면 루프 종료

    # 프레임 출력
    cv2.imshow('Face Registration', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q') or face_registered:
        break

# 캡처 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
