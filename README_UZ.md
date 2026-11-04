# RAC-Fall — boshlang‘ich paket (o‘zbekcha)

Bu arxiv — **birinchi research sprint uchun haqiqiy loyiha asosidir**: ilmiy protokol, dataset auditi, event-level baholash kodi, testlar va YOLO pose smoke-test. Bu hali yiqilishni aniqlaydigan o‘qitilgan model emas.

## MacBook Air M5 (Apple Silicon) da boshlash

Terminalda:

```bash
cd ~/Downloads/RAC-Fall   # arxivni shu joyga ochgan bo‘lsangiz
python3 --version         # 3.11 yoki yangiroq bo‘lsin
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest -q
racfall env-check
```

Keyin webcam yoki foydalanish huquqi bor MP4 bilan faqat pose modelini tekshirish:

```bash
python -m pip install -e '.[vision,dev]'
racfall pose-smoke --source /Users/YOUR_USER/Movies/sample.mp4 --max-frames 50 --device auto
# yoki: racfall pose-smoke --source 0 --max-frames 50 --device auto
```

Birinchi ishga tushganda `yolo26n-pose.pt` modeli internet orqali yuklanishi mumkin. Mac kamera ruxsatini so‘rashi mumkin. `pose-smoke` yiqilishni hali aniqlamaydi, faqat pose inference va FPSni tekshiradi; video saqlamaydi.

Sintetik namunadagi event evaluator:

```bash
racfall eval-events \
  --manifest tests/fixtures/manifest.csv \
  --ground-truth tests/fixtures/gt_events.csv \
  --alerts tests/fixtures/alerts.csv
```

Bu yerda chiqqan sonlar **haqiqiy model natijalari emas**, faqat evaluator to‘g‘ri hisoblayotganini tasdiqlovchi test ma’lumotlari.

## Bosqichlar
1. `docs/RESEARCH_PROPOSAL.md`, `LITERATURE_REVIEW.md` va `DATASET_PROTOCOL.md`ni o‘qing.
2. CAUCAFall/URFD/GMDCSA-24 datasetlarini alohida manbalardan huquq va annotatsiya tekshiruvi bilan oling; ZIP ichida datasetlar yo‘q.
3. CAUCAFall subyektlarini train/val/test bo‘yicha qat’iy ajrating, yiqilish boshlangan vaqtni aniqlang.
4. Codex’ga `docs/CODEX_TASKS.md`dagi **faqat TASK-001** ni topshiring.
5. Keyingi ishlar: M0 posture baseline -> TCN -> context -> RACF -> ilova va blind evaluation.

## Hozirgi tekshiruv
Repo yaratildi va bu muhitda 9 ta sintetik pytest testi o‘tdi. Bu muhit Linux: Apple M5 GPU yoki Ultralytics weight inference haqiqiy tekshirilmagan. Haqiqiy natija uchun o‘z Mac’ingizda diagnostika kerak.
