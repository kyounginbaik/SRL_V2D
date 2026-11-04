# V2D Track 3 최종 제출: 현재 이슈 정리 (2026-11-04 밤, Hansol)

근거: `V2D_FINAL_SUBMISSION_REQUIREMENTS.md` (키트 11-02판 `CONTAINER_SUBMISSION.md`, `tools/verify_container.py`, Kaggle Rules, 대회 웹사이트·FAQ). Kaggle 제출 마감: **11/5 (목) 07:00 KST**.

## A. 마감(07:00 KST) 전에 꼭 할 것

1. **6개 보드 모두 Final Submission을 같은 파일로 지정** (Kaggle UI, 보드당 최대 2개).
   - 지정하지 않으면 보드별 최고점 제출이 카운트됩니다. 지금은 보드마다 최고점 파일이 다릅니다 (RPE = aa8fde7c, MP-SR·SP-SR·CD-O = v33a, AUC/MPPE = 0fca055b 등).
   - 그러면 마감 후 컨테이너가 그 파일들을 **전부** 재현해야 합니다. 파일마다 receipt·체크포인트·메시도 보존해야 합니다.
2. **`Nvidia-V2D-Challenge`(GitHub 사용자 계정, org 아님)를 `kyounginbaik/SRL_V2D` collaborator로 초대했는지 확인.**
   - 레포는 비공개입니다(익명 접근 시 404). 제출 행의 `code_commit_url` = `kyounginbaik/SRL_V2D/commit/5cc5ca51…`입니다.
3. **팀 등록 폼(Google Form) 제출 여부 확인.** Kaggle 규칙 수락과는 별개입니다 (Rules 2.8.d).

## B. 치명적일 수 있는 문제 (주최측 문의 권장)

4. **평가기 해시와 백엔드가 공식과 다름.**
   - FINAL의 `eval_code_sha256 = fdddb348`은 팀 build-v2 규약(소스 바이트 연결 해시)입니다. 공식 `verify_container.py`는 FlashCHORD `flash_chord/evaluation/{__init__,harness,metrics,rollout}.py` 규약으로 다시 계산하므로 **검사가 구조적으로 FAIL**합니다 (공식 값 `b1b3ebe3`).
   - 롤아웃도 공식 FlashCHORD 런타임이 아니라 팀 MuJoCo build-v2 백엔드에서 나왔습니다. Rules 2.9.d와 컨테이너 가이드는 "organiser-provided evaluator"를 전제합니다.
   - Kaggle은 `accepted_*_sha256` 목록이 비어 있어 받아줬지만, 리더보드 순위는 컨테이너 검증 전까지 잠정입니다 (Rules 2.3.c).
   - 제안: 커스텀 evaluator 래퍼를 쓴 제출의 검증 방법과 유효성을 v2d_challenge@nvidia.com에 사전 문의.
5. **커밋 하나(5cc5ca51, 10/23)가 FINAL 전체 코드를 반영하는지.**
   - 그 커밋의 변경은 `evaluator_build_v2_20261023/package_code/` 3개 파일뿐입니다.
   - 11/02–04에 만든 체크포인트·메시 생성 코드(K32/K33 window-beam, teacher/ILC/CEM, distillation, 재구성, splice)가 그 커밋에 없으면 Rules 2.9.a("that commit must reflect the code used") 위반 소지가 있습니다.
   - 이미 올라간 파일의 커밋 URL은 바꿀 수 없습니다. 새로 올릴 파일(ep33 수정본 등)에는 그 코드가 들어간 커밋을 쓰는 방안도 검토해야 합니다.
   - 단, 파일 안 provenance는 20개 에피소드 모두 같아야 합니다(uniform).

5a. **테스트 영상에 사람이 붙인 표시 (Foundational Rules 3.4.b).**
   - 규칙 3.4.b: test 데이터에 "hand labeling or human prediction"을 쓰면 안 된다.
   - ep32 성공본(0.95)의 재구성은 `marks_ep32.json`(평가 영상 58개 프레임, 사람이 표시한 숟가락 bowl 방향)을 비용 항으로 씁니다.
   - ep33의 r6/c18 계열(R6M, R6M2 포함)도 92개 프레임의 사람 표시로 숟가락 roll을 맞췄습니다.
   - 대회 측 Astra 가이드라인은 "human involvement"를 기록하라고 하고, 예시에도 "iterative process with human intervention"이 있습니다. 그래서 숨기지 말고 human-in-the-loop으로 **사실대로 기록**하는 것을 제안합니다.
   - ep33은 독립 에이전트 판정과 비교를 마쳤습니다: 일치 43, 에이전트 판단 불가 29, 반대 20. 6–10초 뒤집힘은 에이전트가 독립적으로 재현했습니다. 보고서 문구는 `EP33_AGENT_SPOON_MARKS.md`에 있습니다.
   - 보고서에 두 annotation의 순서는 쓰지 않습니다(사람 표시 11/03, 에이전트 11/04). 에이전트 "성공률"도 인용하지 않습니다.
   - ep32도 같은 비교를 마쳤습니다(사람 표시 58장): 일치 17, 에이전트 판단 불가 13, 비스듬한 장면이라 에이전트는 앞으로 보고 사람은 측면 방향으로 표시한 것 26, 반대 2. 에이전트 판정만으로도 성공한 K32_RP2가 이전 v14보다 잘 맞습니다(86/88 대 77/88). 문구는 같은 문서에 있습니다.
   - 리더보드 결과를 보고 고른 값(K32_R10 시작 roll, ep33 에피소드별 업로드)도 함께 기록할지 팀이 판단해야 합니다.

## C. 마감 후 (기한은 공개 자료에 없음)

6. **Docker 컨테이너** (`/submission`):
   - 구성: `code/`(커밋 시점 레포 + .git, clean), `kit/`, `verifier_kit/`, `submitted/`(업로드한 파일 + receipt), `weights/`(체크포인트 20개 원본 바이트), `meshes/`, `run.sh verify|reproduce`(명령 하나로 20개 롤아웃 + 패킹, 네트워크 없이)
   - 제출: `docker save | zstd` + sha256 + image digest를, 접근 요청 없이 받을 수 있는 링크와 함께 기술 보고서에 첨부합니다.
   - 확인 필요: 클러스터에서 Docker 빌드가 가능한지, Track 3 `submission.json` 템플릿이 없는 문제, CD-O 파일을 어떻게 지정하는지.
7. **기술 보고서:** 형식·분량·마감·제출 방법이 공개 자료에 없습니다. 보고서에 컨테이너 실행 방법과 run-to-run 분산을 함께 적는 것을 권장합니다. MP-SR 경계 근처 에피소드는 재실행 때 결과가 뒤집힐 수 있습니다.
8. **라이선스:** Kaggle 규칙·FAQ는 Apache-2.0, 웹사이트는 "research license"로 표현이 다릅니다. 서드파티(MANO 비상업 라이선스, DexAgent, SAM3D, FoundationPose 등)가 OSI·상업 사용 조건(Foundational 3.6.c)을 만족하는지도 점검해야 합니다.

## D. ep33 관련 (참고)

- ep33 수정본도 FINAL과 같은 provenance(평가기 해시 cbe866f9/fdddb348, packer/uploader, commit 5cc5ca51)로 만들고 있습니다. 위 4·5번이 해결되는 방식에 같이 영향을 받습니다.
- 공식 FlashCHORD evaluator(160d24f5/b1b3ebe3)로 만든 ep33은 FINAL과 해시가 섞여 strict validation에 실패합니다(이전 FC33 시도).
