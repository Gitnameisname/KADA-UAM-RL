# KADA-UAM-RL
본 저장소는 KADA의 UAM 시뮬레이터를 사용하여 천이과정 강화학습을 진행합니다.

main 브랜치는 최대한 건들지 않고, 별도의 작업을 수행하고자 한다면 develop 브랜치에서 분기점을 만들어서 사용하세요.
Ex. 최철균 과장은 현재 develop 브랜치에서 분기한 personal/ckchoi 브랜치에서 작업중입니다.

각 연구원님께서 작업을 완료하시면 develop 브랜치에 병합 하시기 바랍니다.

# Change history
## 23.12.08
1. 학습과 테스트 모두 동작 안하는것 수정
  - 수정 후 develop 브랜치에 병합

2. 테스트 결과 파일 저장
  - Json 파일로 데이터 저장 | 위치: ./src/results
  - png 파일로 그래프 저장 | 위치: ./src/plots

## 23.12.03
1. 소스코드 구조화
  - 핵심 시뮬레이션 코드는 src(source code) 폴더로 이동
  - 학습 코드와 테스트 코드는 각각 모델로 변경
    - SAC_Transition_Training_Test.ipynb -> Training.py
    - SAC_Transition_test.ipynb -> Test.py
  - 시뮬레이션 코드 변경
    - 시뮬레이션 코드 이름 변경: Transition_Training_8_discrete_simply.py -> Simulator.py
    - 항공기 DB 분리: ./src/DB/aero.json
    - 항공기 DB 불러오기 모델 신규 작성: ./src/loadDB.py
    - 앞으로 항공기 DB 변경 사항 발생 시, 아래와 같이 작업 가능
      - 항공기 모델이 크게 변하지 않는 경우: aero.json 변경
      - 항공기 모델이 크게 변하는 경우: DB 폴더에 새로 Json DB 반입 -> 적용하는 DB 변경
        ```python
        # Simulator.py - line: 17
        # 아래의 DB 이름을 새로운 DB 이름으로 변경
        self.set_DB("aero.json")
        ```
  - 학습과 테스트는 `Playground.ipynb`에서 수행

## 23.08.06
1. 원본 코드와 별개로, 주석 일부를 삭제하고 보상 계산 방법을 변경한 시뮬레이션 파일 생성(CK.Choi)
   - Transition_Training_8_discrete_simple.py
2. 학습 코드 새로 생성 및 수정(CK.Choi)
   - SAC_Transition_Training_Test_ckchoi.ipynb
