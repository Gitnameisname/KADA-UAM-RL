# KADA-UAM-RL
본 저장소는 KADA의 UAM 시뮬레이터를 사용하여 천이과정 강화학습을 진행합니다.

main 브랜치는 최대한 건들지 않고, 별도의 작업을 수행하고자 한다면 develop 브랜치에서 분기점을 만들어서 사용하세요.
Ex. 최철균 과장은 현재 develop 브랜치에서 분기한 personal/ckchoi 브랜치에서 작업중입니다.

각 연구원님께서 작업을 완료하시면 develop 브랜치에 병합 하시기 바랍니다.

# 코드 변경점
1. 원본 코드와 별개로, 주석 일부를 삭제하고 보상 계산 방법을 변경한 시뮬레이션 파일 생성(CK.Choi)
   - Transition_Training_8_discrete_simple.py
2. 학습 코드 새로 생성 및 수정(CK.Choi)
   - SAC_Transition_Training_Test_ckchoi.ipynb
