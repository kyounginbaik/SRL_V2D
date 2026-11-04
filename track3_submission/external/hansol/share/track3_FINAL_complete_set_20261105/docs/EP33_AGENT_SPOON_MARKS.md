# ep33 spoon front/back: independent agent annotation vs the user marks (2026-11-04, 22:35 KST)

## What was done
- Crops: `ep33_annot_crops.py` on the ep33 evaluation RGB, every 2nd frame f0–f346 (174 crops around the SAM2 spoon mask, no reconstruction drawn, no previous marks attached). Server: `/data1/hansol/ep33_20261102/agentmarks/crops`.
- Annotators: three fresh Claude subagents (general-purpose, no conversation context, never shown the user marks, told to read only the crops), frames 0–114 / 116–230 / 232–346. Prompt: `agentmarks_prompt.md` (same folder). Output per frame: `out` / `in` / edge-on `[dx, dy]` / `unsure`, plus confidence and a short reason.
- Cost: about 282k subagent tokens, about 4 minutes wall-clock.
- Scoring: `spoon_annot_roll2.py --report-only --face-thr 0.25` (does a reconstruction's spoon agree with each mark), for r6 (= c18 = v14 in these frames) and H5r16 (FINAL's ep33 target).

## Results
| marks | r6 / c18 (flipped lineage) | H5r16 (FINAL) |
|---|---|---|
| agent (135 non-unsure) | **123 / 135** (out 117/118, edge 6/17) | 110 / 135 (out 98/118, edge 12/17) |
| user (92) | 80 / 92 | 61 / 92 |

- Direct agent vs user on the 92 user-marked frames:
  - **out:** 40 agree, 0 contradict.
  - **in:** the agent called none; the 14 user "in" frames (f144–152, f204–218) are "unsure" for the agent.
  - **edge-on:** 14 both edge-on, but the agent's arrow points the opposite way in f116–118, f186–190 and f252–262 (low confidence).
- Frames where r6 and H5r16 differ:
  - **f120–136 and f170–198 (6–10 s):** the agent prefers r6 in all 19 frames, 6 of them at high confidence. This is the flip that separates the two lineages.
  - **f252–262 (12.6–13.1 s):** the agent's low-confidence edge-on arrows fit H5r16; the user marks fit r6.

## Breakdown on the 92 frames that carry a final (human-reviewed) mark
| case | frames |
|---|---|
| agent and final mark agree | 43 (out 40, edge-on same direction 3) |
| agent abstained (bowl hidden by the cup or blurred), final mark set by the human | 29 (14 back, 12 edge-on, 3 front) |
| agent and final mark disagree, human decision kept | 20 (11 edge-on opposite direction, 8 edge-on vs front, 1 back vs edge-on) |

## Wording for the report (Astra "human involvement" item)
**EN.** The front/back orientation of the spoon bowl in ep33 was set human-in-the-loop. The agent generated per-frame crops and silhouette overlays, and an independent agent instance (prompt attached) labelled each frame as front, back, edge-on with a direction, or unsure. A human reviewed these and fixed the final labels for 92 frames: 43 agree with the agent, 29 are frames where the agent abstained (the bowl is hidden by the cup), and 20 are frames where the agent and the human disagree, mostly edge-on views. The final labels enter the spoon-roll fit as a cost term. Without human input, the agent independently reproduces the decisive flip of the spoon at 6–10 s that separates our reconstruction from the earlier one.

**KR.** ep33 숟가락 bowl의 앞뒤 방향은 human-in-the-loop으로 정했다. 에이전트가 프레임별 crop과 실루엣 오버레이를 만들고, 독립된 에이전트(프롬프트 첨부)가 각 프레임을 앞 / 뒤 / 측면(방향 포함) / 판단 불가로 판정했다. 사람이 이를 검토해 92개 프레임의 최종 표시를 정했다. 에이전트와 일치 43, 에이전트가 판단 불가(컵에 가림) 29, 서로 반대 20(대부분 측면)이다. 최종 표시는 숟가락 roll fit의 비용 항으로 들어간다. 사람 입력 없이도 에이전트는 이전 재구성과 우리 재구성을 가르는 6–10초 뒤집힘 판단을 독립적으로 재현했다.

Notes for whoever writes the report:
- State no order between the two annotations. The human marks were made on 11/03 and the agent annotation on 11/04, and the files and logs show this.
- Do not quote an agent "success rate". The only reference is the human labels, and the agent abstains on frames where the bowl is not visible.

## ep32 (same method, 2026-11-04 23:15 KST)
- Crops: every 3rd frame f1–f256 plus f140–f209 to line up with the human marks (110 crops, no reconstruction, no marks). Three fresh subagents, same prompt. About 257k subagent tokens. Server: `/data1/hansol/ep33_20261102/agentmarks32/`.
- Human marks: `marks_ep32.json` from the shared ep32 package (58 frames, f40–f209).
- Reconstructions: K32_RP2 (the successful T32c_b1, MP-SR 0.95) and v14_g17 (the earlier one).

| marks | K32_RP2 | v14_g17 |
|---|---|---|
| agent (88 non-unsure) | **86 / 88** | 77 / 88 |
| human (58) | 57 / 58 | 32 / 58 |

- All 9 frames where the agent separates the two reconstructions favour K32_RP2:
  - f70, f130 and f184 (front)
  - f140 and f143 (back)
  - f173–182 (edge-on)
- On the 58 human-marked frames:

| case | frames |
|---|---|
| agree | 17 (front 10, back 3, edge-on same direction 4) |
| agent abstained (bowl in the cup), human decided | 13 |
| human edge-on, agent front | 26 (oblique views; K32_RP2 satisfies both) |
| edge-on, opposite direction | 2 (f118, f121) |

- Reading: without human input, the agent also prefers the ep32 reconstruction that passed, but more weakly than the human marks do. Its main extra evidence is the 7–9 s segment. It does not resolve most edge-on directions.

**ep32 wording (EN).** As for ep33: the human set the final labels for 58 frames after reviewing an independent agent annotation (prompt attached). 17 agree with the agent, 13 are frames where the agent abstained (the bowl is hidden by the cup), 26 are oblique views that the agent called front and the human labelled edge-on with a direction, and 2 disagree. On its own, the agent annotation also prefers the final reconstruction over the earlier one (86/88 vs 77/88 frames consistent).

**ep32 문구 (KR).** ep33과 같은 방식이다. 독립 에이전트 판정(프롬프트 첨부)을 검토한 뒤 사람이 58개 프레임의 최종 표시를 정했다. 에이전트와 일치 17, 에이전트 판단 불가(컵에 가림) 13, 비스듬한 장면이라 에이전트는 앞으로 보고 사람은 측면 방향으로 표시한 것 26, 서로 반대 2이다. 에이전트 판정만 써도 최종 재구성이 이전 재구성보다 잘 맞는다(일치 86/88 대 77/88).

## Reading
- Without any human input, the agent reproduces the main front/back decision of the r6/c18 reconstruction (the flipped spoon at 6–10 s).
- It does not confirm the user's "in" frames (it abstains) or the edge-on direction at 12.6–13.1 s.
- Still true: the submitted r6/c18 lineage was fitted with the user marks. An agent-only reconstruction would need the roll refits re-run with `agent_marks.json`, which changes f236–290 at least.
