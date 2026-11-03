# 캐릭터 LoCon 학습 설정

| 파일 | 내용 |
|---|---|
| `character_locon_1024_b4.toml` | Kohya 데이터셋 설정. 캐릭터 8명을 subset으로 나누고, 데이터가 적은 캐릭터일수록 repeat를 높여 노출을 맞춤 (5~13) |
| `launch_character_locon_b4.sh` | 서버(L40S) 학습 실행 스크립트 |

이 두 파일은 **8명을 한 모델로 학습한 통합 캐릭터 LoCon**의 설정입니다.

현재 Studio가 쓰는 캐릭터별 모델은 같은 캐릭터 subset과 repeat를 캐릭터마다 따로 학습했습니다. 하이퍼파라미터는 같고, 학습 step이 줄어든 만큼 warmup만 23 step으로 다릅니다. 캐릭터별 학습의 원본 실행 명령은 남아 있지 않아, 각 가중치의 학습 metadata로 설정을 확인했습니다 ([실험 기록 ⑤ 단계](../../docs/experiments.md)).

경로는 `./data/`, `$HOME/webtoon_lora/` 기준으로 바꿔 두었습니다. 학습 데이터는 저장소에 포함하지 않습니다.
