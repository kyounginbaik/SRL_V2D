# 코드 제출본 정리 (2026-11-05, 최종 파일 9bc322af)

## 대상 파일
`/data1/hansol/spoon3537/rt/packages/track3_FINALc_ep36s4/all_20_episodes/track3_submission.parquet`
- sha256 `9bc322af…`, CD-O `a03230e5…`
- 구성은 다음과 같습니다.
  - FINAL 0fca055b
  - ep32 T32c_b1과 ep33 R6M2로 교체 → 8b6fa2b9. MP-SR 1.0, AUC 0.92086, MPPE 1.40441, RPE 1.50325
  - **ep36 s4**로 교체 → 9bc322af. AUC 0.92269, MPPE 1.38958. RPE는 그대로입니다(단일 물체 에피소드). MP-SR은 1.0으로 예상합니다.
- 비교한 안 B(s4 + ep37 refit, 452d4a7a)는 MPPE 1.39543으로 더 나빠서 쓰지 않았습니다.

## 제출본에 들어 있는 것
- **경로 대응:** 최상위 폴더는 `/data1/SRL_V2D`, `external/hansol`은 `/data1/hansol`, `external/hansol_home`은 `/home/hansol`에 대응합니다.
- **코드:** 코드와 작은 설정 파일만 넣었습니다. weights, npz, 영상은 넣지 않았습니다. secret 패턴 검사는 4번 했고 모두 0건입니다.
  - DexAgent upstream `e29bdad`(수정 없음)
  - **SRL DexAgent 수정본:** floating Sharpa 포트, actor, ILC, 증류
  - evaluator build v2, campaign_v16/v22, GT-free 재구성
  - ep32 코드(T32c_b1), ep33 코드(R6M2: `ep33_20261102`, `variants_20261030`, `retarget33`, `teacher33`)
  - **ep36 s4 코드:** `spoon3537`의 `code`, `rt/code`, `am36` agent 표시
- **`track3_submission/` 아래 문서와 파일:**
  - `provenance/FINAL_9bc322af_episodes.tsv`: 20개 에피소드, 9,120행. 각 에피소드의 checkpoint와 절대경로로 쓴 component·candidate 위치가 있습니다. ep36은 `5d5738b7`입니다.
  - `container/`: `submission.json`(sha 9bc322af, AUC 57997980, MPPE 57997979), `run.sh`, `Dockerfile`, `collect_weights.sh`(Slurm에서 실행)
  - `docs/`:
    - `HUMAN_IN_THE_LOOP.md`: ep32/33은 사람 표시, ep36은 agent 표시에 사용자의 렌더 검토
    - `TRIED_AND_REJECTED.md`: B, ep37, ep35, ep3 fix/e3s, ep36 s1d/s2/s3/s5/s6/A, RGB와 stereo 카메라 결과가 어긋난 점
    - `TECH_REPORT_DRAFT.md`, `PIPELINE.md`, `DEXAGENT_MODIFICATIONS.md`, `BUILD_CONTAINER.md`, `NOTICE.md`
  - `LICENSE`(Apache-2.0)
- **하지 않은 것:** 커밋, push, Kaggle 업로드, `collect_weights.sh` 실행

## 팀이 정해야 할 것
1. **커밋 URL**
   - 업로드된 행의 `code_commit_url`은 `5cc5ca51`입니다. 이 커밋에는 최종 파일을 만든 코드가 거의 없습니다.
   - A안: 이 overlay를 커밋하고, 이 커밋 URL로 다시 pack합니다. 마감이 지났으므로 주최측과 상의해야 합니다.
   - B안: `code/`는 `5cc5ca51`로 두고, overlay를 보충 코드로 내면서 보고서에 밝힙니다.
2. **6개 보드 Final을 9bc322af 하나로 통일할지**
   - 통일하면 container가 재현할 파일이 하나입니다.
   - 아직 receipt가 없는 보드: MP-SR, SP-SR, RPE. RPE는 8b6fa2b9와 같은 점수입니다.
3. **CD-O:** 메시는 FINAL과 같습니다. CD-O 파일은 `a03230e5`입니다.
4. **evaluator 해시:** `eval_code_sha256 fdddb348`은 팀 규약이라 kit verifier에서 FAIL이 확실합니다. 설명이 필요합니다.
5. **사람 개입 공개**
   - ep32/33의 사람 표시는 Kaggle 규칙 3.4.b와 관련됩니다.
   - ep36은 agent 표시이고, 사람은 렌더를 보고 고칠 구간을 짚었습니다.
   - 리더보드를 참고해 정한 값도 있습니다.
   - `HUMAN_IN_THE_LOOP.md` 문구를 보고서에 넣고, 주최측에 문의할지 정해야 합니다.
   - ep36 subagent의 토큰 사용량은 기록이 없어 TODO로 남아 있습니다.

## 마감 뒤 할 일
1. **커밋과 push (팀원):**
   - `kyounginbaik/SRL_V2D`의 `5cc5ca51`(evaluator-build-v2)을 checkout하고, 이 폴더를 repo 루트에 복사해 검토한 뒤 커밋하고 push합니다.
   - GitHub 사용자 `Nvidia-V2D-Challenge`에게 repo를 공유합니다.
2. **Weights (Slurm):**
   ```bash
   sbatch -p debug -c 2 --mem=8G -t 01:00:00 --wrap "bash track3_submission/container/collect_weights.sh <BUILD>/submission"
   ```
   20개가 모두 `ok`로 나와야 합니다.
3. **Container:** `docs/BUILD_CONTAINER.md` 순서대로 `build` → `verify` → `reproduce --episodes 1`
4. **전달:** `docker save | zstd`로 만든 이미지를 Drive에 올리고, sha256과 digest를 기록합니다. 기술 보고서와 함께 주최측에 보냅니다. 이 단계는 팀이 합니다.
