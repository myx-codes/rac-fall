# Risk-Adaptive Context-Aware Fall Verification

Research & Development Master Plan — Version 1.0

Loyihani boshlashga tayyor

4-noyabr, 2026

Ha, shu loyihani boshlashni tavsiya qilaman. Ammo uni shunchaki Fall Detection dasturi emas, ilmiy jihatdan asoslangan va tajriba orqali tekshiriladigan yangi AI tizimi sifatida quramiz.

Eng muhim qarorimiz: dastlab modelning o‘zini emas, butun tadqiqot metodologiyasini loyihalashtiramiz. Shunda 2–3 oy kod yozib, oxirida modelimiz mavjud tadqiqotlardan farq qilmasligini bilib qolmaymiz.

## 1. Loyiha pasporti

| Bo‘lim          | Qarorimiz                                            |
| --------------- | ---------------------------------------------------- |
| Project Name    | RAC-Fall                                             |
| Full Name       | Risk-Adaptive Context-Aware Fall Verification        |
| Turi            | Independent AI Research + Software Engineering       |
| Maqsad          | Yiqilishni aniqlashda false alarmni kamaytirish      |
| Asosiy yangilik | Risk-adaptive context verification                   |
| Kamera          | MacBook webcam / MP4                                 |
| Kompyuter       | MacBook Air M5, 24 GB                                |
| AI Framework    | PyTorch + YOLO Pose                                  |
| Dasturlash      | Python + FastAPI + React                             |
| Privacy         | Local inference, minimal data retention              |
| Loyiha natijasi | Model + real-time dastur + ilmiy tajribalar + GitHub |
| Ishchi muddat   | 16 hafta                                             |

REJALASHTIRILGAN MUDDAT

# 16 hafta

5-noyabr 2026 – 24-fevral 2027

Kuniga 4 soat, haftasiga 6 kun — jami taxminan 384 soat.

Bu vaqt ajratishingiz bo‘yicha hozircha qabul qilingan taxmin, qat'iy deadline emas. Dataset va tadqiqot natijalariga qarab muddat o‘zgarishi mumkin.

## 2. Loyihaning aniq ilmiy maqsadi

Biz quyidagi muammoni hal qilishga harakat qilamiz:

Odamning yiqilishi bilan unga o‘xshash kundalik harakatlarni ajratishda bir xil threshold ishlatish o‘rniga, harakatning dinamikasi, atrof-muhit, pose sifati va model noaniqligiga asoslangan adaptiv tekshiruv yaratish.

Masalan:

| Vaziyat                   | Tizim nimani tahlil qiladi?                                 |
| ------------------------- | ----------------------------------------------------------- |
| Odam stulga o'tirdi       | Stul mavjudligi, pastga tushish tezligi, nazoratli harakat  |
| Odam polga egildi         | Tananing holati, tezligi, keyingi tiklanish                 |
| Odam divanga yotdi        | Divan bilan munosabat, harakat ketma-ketligi                |
| Odam to‘satdan yiqildi    | Tez vertikal siljish, muvozanat yo‘qotilishi, keyingi holat |
| Odam qisman ko‘rinmayapti | Pose confidence va yetishmayotgan ma'lumot                  |

Biz yaratadigan tizim uch xil javob beradi:

Normal

Xavf dalili yetarli emas

Uncertain

Tekshiruv talab qilinadi

Fall Alert

Yiqilish ehtimoli yuqori

Muhim: biror obyekt yoki pose yaxshi ko‘rinmasa, model buni avtomatik ravishda xavfsiz deb hisoblamaydi. Noaniqlik ortganda, dalillar qayta tekshiriladi yoki operatorga Uncertain holati ko‘rsatiladi.

## 3. Tanlagan arxitekturamiz

RGB Video Input

Webcam / MP4 · Local processing

Pose & Tracking

Human skeleton, person ID, movement

Scene Context

Chair, bed, sofa, floor geometry

Temporal Motion Model (TCN)

Movement history · Fall probability · Pose quality

RACF — Proposed Verification Module

Risk estimation · Context fusion · Confidence calibration · Adaptive temporal confirmation

Biz tadqiq qiladigan asosiy modul

Event Engine + Privacy Layer

Normal / Uncertain / Alert · Local logs

RACF moduli faqat qoidalar to‘plami bo‘lib qolmasligi kerak. U kichik, o‘qitiladigan risk-verification modeli yoki kalibrlangan scoring mexanizmi bo‘ladi. Qaysi variant samaraliroq ekanini baseline tajribalari bilan aniqlaymiz.

Ushbu arxitekturadagi qismlarning ko‘pi avvalgi ilmiy ishlarda mavjud. Masalan, Xiao va boshqalar 2026-yilda RGB–Pose TCN va alarm verification'ni o‘rgangan; natijada false alarm kamaygan, lekin recall ham pasaygan. Biz aynan shu muvozanatni yaxshilashga urinib, faqat barcha modullarni birlashtirishni ilmiy yangilik deb da'vo qilmaymiz.&#x20;

MDPI



## 4. Development Roadmap — 16 hafta

Har bir bosqichda aniq natija va undan keyingi bosqichga o'tish sharti bo‘ladi.

01

Research Foundation & Dataset Audit

1–2-haftalar

5–18 noyabr

- Asosiy maqolalarni qayta ko‘rib chiqish va research gap'ni aniqlash
- Datasetlarni yuklab olish, litsenziya va annotatsiyalarni tekshirish
- Research questions, hypothesis, baseline va evaluation protocol'ni yozish

Natija: Research Proposal v1.0, Dataset Manifest, Evaluation Protocol

&#x20;Qabul mezoni: Datasetlar ochiladi, event label va subject ID lar tekshiriladi.

02

Video Engine & Pose Baseline

3–4-haftalar

19 noyabr – 2 dekabr

- Mac webcam va MP4 video pipeline
- YOLO Pose, person tracking, skeleton visualization
- Oddiy posture/velocity-based fall baseline
- Real-time FPS va latency'ni birinchi o‘lchash

Natija: Ishlaydigan webcam/MP4 demo va Baseline A

&#x20;Qabul mezoni: Videoni uzluksiz qayta ishlaydi, FPS o‘lchanadi, baseline natijalari chiqadi.

03

Temporal Fall Recognition

5–6-haftalar

3–16 dekabr

- Pose normalization va movement features
- TCN (yoki GRU) training va validation
- Subject-independent split va event evaluation
- Baseline A va B taqqoslash

Natija: O‘qitilgan temporal model va benchmark

&#x20;Qabul mezoni: Model mustaqil validatsiyada posture-only baseline bilan taqqoslangan.

04

Context & Risk-Adaptive Verification

7–9-haftalar

17 dekabr – 6 yanvar

- Bed/chair/sofa va scene geometry extraction
- Person-object relations, pose quality va uncertainty features
- RACF modulining birinchi versiyasi
- Fixed threshold, fixed verification va adaptive verification ablation

Natija: Proposed RACF v1 va ablation results

&#x20;Qabul mezoni: Context va adaptive modullar hissasi alohida o‘lchangan.

05

Privacy-Preserving Inference

10–11-haftalar

7–20 yanvar

- Raw video retention'ni o‘chirib, memory-only processing
- Skeleton-only va RGB-assisted variantlarni solishtirish
- Local event metadata, retention controls, privacy threat model
- Shaxsni qayta aniqlash xavfi bo‘yicha dastlabki audit

Natija: Privacy mode va privacy–accuracy comparison

&#x20;Qabul mezoni: Odatiy ish rejimida raw video/facial data saqlanmasligi test bilan tasdiqlangan.

06

Independent Evaluation & Robustness

12–14-haftalar

21 yanvar – 10 fevral

- Locked external datasetdagi yakuniy blind test
- Turli kamera burchagi, past yorug‘lik, occlusion va normal harakatlar
- Long-duration false-alarm test
- Bootstrap confidence intervals, failure-case analysis

Natija: Final metrics, ablation tables va error analysis

&#x20;Qabul mezoni: Natijalar qayta ishlatiladigan skriptlar bilan reproduksiya qilinadi.

07

Application, Optimization & Documentation

15–16-haftalar

11–24 fevral

- React monitoring dashboard va event review
- Mac M5 FPS/latency/memory optimization
- Automated tests, reproducible training va inference
- GitHub README, demo video, technical report

Natija: Complete research prototype va research report

&#x20;Qabul mezoni: Webcam/MP4 demo, model benchmark, privacy audit va hujjatlar tayyor.

Bu ishlarning hammasini bir vaqtda boshlamaymiz. Ayniqsa dashboard qurishga shoshilmaslik kerak: tadqiqot modelimiz ishlamaguncha chiroyli interfeys asosiy muammoni hal qilmaydi.

## 5. Dataset strategiyasi

Tekshirilgan manbalardan boshlang‘ich uchta datasetni tanlaymiz.

| Dataset                                                                                 | Ishlatilishi                              | Sabab                                                                           |
| --------------------------------------------------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| [CAUCAFall](https://data.mendeley.com/datasets/7w7fccy7ky?utm_source=chatgpt.com)       | Training va ichki validation/test         | 10 subyekt, turli yorug‘lik va kameraga oid sharoitlar, video va annotatsiyalar |
| [UR Fall Detection](https://fenix.ur.edu.pl/~mkepski/ds/uf.html?utm_source=chatgpt.com) | Qo‘shimcha external development benchmark | 30 fall va 40 kundalik faoliyat ketma-ketligi                                   |
| [GMDCSA-24](https://doi.org/10.5281/zenodo.13354453?utm_source=chatgpt.com)             | Yakuniy blind test                        | 81 fall va 79 ADL video; yotish va push-up kabi chalg‘ituvchi harakatlar        |

Manbalar:&#x20;

Mendeley Data+2



Dataset protokoli:

- CAUCAFall'da subyektlar bo‘yicha ajratilgan training, validation va internal test qilamiz. Dastlab 6/2/2 subyekt ajratilishi mumkin; keyin group cross-validation yordamida natijaning barqarorligini tekshiramiz.
- URFD'ni yordamchi benchmark sifatida ishlatamiz. U bilan parametr sozlansa, uni sof blind test deb atamaymiz.
- GMDCSA-24'ni final test uchun yopiq saqlaymiz. Model tanlash, threshold sozlash yoki qayta training qilishda bu datasetdan foydalanmaymiz.

Context annotation uchun ham alohida ish qilamiz. Public datasetlardagi fall/non-fall yorliqlari avtomatik ravishda chair, bed, floor yoki uncertainty ground-truth mavjudligini anglatmaydi. Zarur bo‘lsa training qismida qo‘shimcha scene annotations yaratamiz.

Yana bitta kerakli dataset: uzoq davom etadigan normal harakat videolari. Buning uchun rozilik bilan yozilgan shaxsiy test videolari va foydalanishga ruxsat etilgan boshqa ADL materiallarini yig‘amiz. Dastlab 10 soatlik sinov, keyin 30 soatga yaqin mustaqil normal-video tekshiruvini maqsad qilamiz.

## 6. Baseline va Proposed modelni qanday taqqoslaymiz?

Tadqiqotimizdagi eng muhim jadval shu bo‘ladi.

| Model                              | Pose | Temporal | Context | Adaptive |
| ---------------------------------- | ---- | -------- | ------- | -------- |
| M0 — Posture rules                 | ✓    | —        | —       | —        |
| M1 — Pose-TCN                      | ✓    | ✓        | —       | —        |
| M2 — Pose-TCN + fixed verification | ✓    | ✓        | —       | —        |
| M3 — Context-aware TCN             | ✓    | ✓        | ✓       | —        |
| M4 — RACF Proposed                 | ✓    | ✓        | ✓       | ✓        |

M2 fixed verification modeli bilan M4 adaptive verification modelini bir xil dataset va bir xil false-alarm sharoitida solishtiramiz.

Shunda natijadagi farq haqiqatdan ham bizning risk-adaptive modulimiz hisobiga yuz berganini tekshirish mumkin.

### Asosiy baholash mezonlari

Event Recall

# ≥90%

Haqiqiy yiqilishlarni aniqlash maqsadi

False Alarms

# ≤0.1/h

Mustaqil normal videodagi maqsad

Median Detection Delay

# ≤2 s

Yiqilishdan ogohlantirishgacha

Live Performance

# ≥10 FPS

MacBook'da 720p video

Bu raqamlar ilmiy-texnik maqsadlar, kafolatlangan natijalar emas. Recall, false alarms/hour va delay birgalikda tekshiriladi.

Har bir model uchun precision, recall, F1, AUPRC, calibration, detection delay, FPS va memory usage ham yoziladi.

Barcha natijalar event-level baholanadi: bir yiqilish davomida ketma-ket berilgan 30 ta alarm 30 ta alohida to‘g‘ri aniqlash deb hisoblanmaydi. Hodisa chegaralari va moslashtirish qoidalarini tajribadan oldin belgilaymiz.

## 7. MacBook M5 uchun texnik stack

| Qism               | Texnologiya                              |
| ------------------ | ---------------------------------------- |
| Development        | Codex + VS Code                          |
| Python environment | Python 3.11, venv yoki uv                |
| Deep Learning      | PyTorch                                  |
| GPU                | Apple MPS                                |
| Detection / Pose   | YOLO26n-Pose, pretrained                 |
| Tracking           | ByteTrack yoki BoT-SORT                  |
| Temporal Model     | Lightweight TCN                          |
| Context Detection  | Object detector + scene geometry         |
| Risk Verification  | Calibrated classifier / lightweight MLP  |
| Backend            | FastAPI                                  |
| Frontend           | React + TypeScript                       |
| Database           | SQLite                                   |
| Experiments        | MLflow yoki mahalliy experiment logs     |
| Tests              | pytest + reproducible evaluation scripts |

YOLO26n-Pose 17 ta COCO skeleton nuqtasini chiqaradi va Ultralytics tracking'ni qo‘llab-quvvatlaydi. PyTorch MPS esa Apple Silicon'da GPU acceleration imkonini beradi. Ammo MPS'da ayrim operatsiyalarning qo‘llab-quvvatlanishi cheklangan bo‘lishi mumkin, shuning uchun birinchi haftalarda smoke test zarur.&#x20;

GitHub+2



Keyinchalik commercial foydalanish ehtimoli bo‘lsa, model weight va software litsenziyalarini ham tekshiramiz.

## 8. Codex bilan ishlash strategiyamiz

Codex'ga butun loyihani bir martada yaratishni topshirmaymiz. Quyidagi qoida bo‘yicha harakat qilamiz:

1\. Specification

Vazifa, input/output, test mezonlari belgilanadi.

2\. Codex Implementation

Faqat shu bosqichning kodi va testlari yaratiladi.

3\. Experiment & Evaluation

Model haqiqiy ma'lumotda sinovdan o‘tkaziladi.

4\. Review & Commit

Natijalar tekshiriladi, o‘zgarishlar commit qilinadi va keyingi vazifa ochiladi.

### Project folder structure

```
RAC-Fall/
├── docs/
│   ├── PRD.md
│   ├── RESEARCH_PROPOSAL.md
│   ├── LITERATURE_REVIEW.md
│   ├── ARCHITECTURE.md
│   ├── DATASET_PROTOCOL.md
│   ├── EVALUATION_PROTOCOL.md
│   └── ROADMAP.md
├── configs/
├── data/
│   ├── raw/
│   ├── processed/
│   └── manifests/
├── src/
│   ├── video/
│   ├── pose/
│   ├── tracking/
│   ├── temporal/
│   ├── context/
│   ├── racf/
│   └── privacy/
├── experiments/
├── evaluation/
├── backend/
├── frontend/
├── tests/
├── outputs/
└── README.md
```

Datasetlar, video yozuvlar va model weight fayllarini Git repository'ga tasodifan yuklamaslik uchun `.gitignore` va data retention qoidalarini boshidan sozlaymiz.

## 9. Birinchi 7 kunlik aniq topshiriqlar

Bu bizning eng birinchi sprintimiz bo‘ladi.

1

KUN

Research Proposal

5-noyabr, payshanba

Problem statement, research gap, 4 research question va asosiy gipotezalarni yozish.

&#x20;RESEARCH_PROPOSAL.md

2

KUN

Related Work Matrix

6-noyabr, juma

Eng yaqin 6–10 maqolaning arxitektura, dataset, metrika, kamchilik va reproducibility jihatlarini taqqoslash.

&#x20;LITERATURE_REVIEW\.md

3

KUN

Dataset Audit

7-noyabr, shanba

CAUCAFall, URFD va GMDCSA-24 availability, hajm, annotatsiya, subject, FPS va litsenziyasini tekshirish.

&#x20;DATASET_PROTOCOL.md

4

KUN

Evaluation Protocol

8-noyabr, yakshanba

Event matching, false alarms/hour, recall, threshold tuning, blind-test va leakage qoidalarini belgilash.

&#x20;EVALUATION_PROTOCOL.md

5

KUN

Architecture Design

9-noyabr, dushanba

Pose, tracking, TCN, context va RACF interfeyslarini loyihalash; latency va privacy talablarini yozish.

&#x20;ARCHITECTURE.md

6

KUN

Environment & Feasibility

10-noyabr, seshanba

Repository tuzilishi, Python/PyTorch MPS testlari va pretrained pose model bilan bitta test videoda inference.

&#x20;Ishlaydigan smoke test va environment report

7

KUN

Sprint Review

11-noyabr, chorshanba

Research va texnik natijalarni ko‘rib chiqish, hal etilmagan risklarni belgilash, keyingi sprintni tasdiqlash.

&#x20;SPRINT_01_REPORT.md

Bu yerda yetti kun — kalendar bo‘yicha reja. Sizga 6 kunlik ish haftasi qulay bo‘lsa, Sprint Review'ni navbatdagi ish kuniga ko‘chiramiz.

## 10. Qachon loyiha muvaffaqiyatli hisoblanadi?

Minimum research success

Proposed RACF model bir xil evaluation protokolida fixed-verification baseline'ga nisbatan yolg‘on ogohlantirishlarni kamaytirishi va recall'dagi salbiy o‘zgarish aniq ko‘rsatilgan bo‘lishi kerak. Tajriba natijalari oldindan belgilanmagan ijobiy raqamlarga moslashtirilmaydi.

Minimum engineering success

MacBook kamerasi yoki MP4 video bilan end-to-end inference, vaqt belgilari, event logs, alert va testlar ishlashi kerak. Internet talab qilinmaydi. Odatiy privacy rejimida raw video saqlanmaydi.

Research publication readiness

Mustaqil blind test, ablation study, uncertainty/calibration analysis, confidence interval, mavjud tadqiqotlar bilan metodologik taqqoslash va reproduksiya qilinadigan kod bo‘lishi kerak.

Xavfsizlik chegarasi

Bu tadqiqot prototipi. Natijalar haqiqiy tibbiy yoki favqulodda xavfsizlik tizimi sifatida tasdiqlanmaguncha, odamning xavfsizligini faqat shu AI'ga ishonib topshirib bo‘lmaydi.