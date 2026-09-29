# Roadmap

Hedef: projeyi bir **prototip** olarak, gerçek görüntü ve kalıcı veriyle çalışan bir QC sistemine taşımak.
Her faz bağımsız PR/sürüm olarak teslim edilir. Sayılar README'ye elle yazılmaz, `docs/data/` JSON'larından üretilir.

## Kararlar

| Konu | Karar |
|------|-------|
| Hedef | Prototip. Gerçek kamera entegrasyonu kapsam dışı; video döngüsüyle sahte kamera yeterli |
| GPU | Mevcut. Model eğitimi ve RAG değerlendirmesi yerelde/GPU'da çalıştırılır |
| Veritabanı | Neon Postgres, **bu projeye ayrı bir Neon projesi** (`visual-qc`). Başka projelerle (ör. VocabMaster) proje, veritabanı, rol ve bağlantı dizesi paylaşılmaz. Yerelde SQLite |
| Deploy | Vercel yalnızca demo (hafif ONNX inference). Tam sistem Docker ile |

## Faz 1: Sağlamlaştırma (tamam)
- [x] `main.py` modüllere bölündü (`hmi_page`, `camera`, `report`)
- [x] `coverage` ölçümü ve CI eşiği; `ruff`; `pre-commit`; Dependabot; mypy
- [x] README/hiring-summary sayıları `tests/test_docs_consistency.py` ile JSON'a karşı doğrulanır (üretim yerine sapma koruması)
- [x] `line_sim.snapshot` maliyeti çalışma süresiyle artıyordu (7 günlük hatta ~330 ms/istek); OK sayısı blok önbelleğine alındı, ısınmış istek ~0.4 ms

## Faz 2: Gerçek model, gerçek görüntü (boru hattı CPU'da doğrulandı; gerçek eğitim GPU'da)

Hazır: `analysis/run_dl_case_study.py` (ResNet18 / EfficientNet-B0, RF ile aynı holdout, sıcaklık ölçekleme, ECE, bootstrap GA, isteğe bağlı K-fold CV), `analysis/dl/` (test edilmiş NumPy/SciPy yardımcıları). Torch kısmı bulut konteynerinde çalıştırılamadığı için ilk GPU çalıştırması bir doğrulama adımıdır. Kalan: Grad-CAM, ONNX, dashboard'a gerçek görüntü, `/api/classify`.
- ResNet18 / EfficientNet-B0 fine-tune; aynı 80/20 holdout ve metrik tablosuyla Random Forest ile karşılaştırma
- Temperature scaling, reliability diagram, kalibre inceleme kuyruğu
- 5-fold stratified CV ve bootstrap güven aralıkları
- Grad-CAM; scratches↔inclusion karışıklığının incelenmesi
- Dashboard'da prosedürel doku yerine gerçek NEU-CLS örnekleri
- ONNX export ve `POST /api/classify` (yükle, sınıflandır, güven + Grad-CAM)

Çıkış kriteri: karşılaştırma tablosu README'de; CV ve kalibrasyon sonuçları belgelenmiş.

## Faz 3: Kalıcılık ve operatör akışı (tamam)

Hazır: `store.py` (Postgres/SQLite, `DATABASE_URL`), `review.py` (`/api/review-queue`, `/api/review`, `/api/review/export.csv`), HMI'de Review Queue paneli, ayrı Neon projesi `visual-qc`. Postgres SQL'i Neon'da doğrulandı; psycopg bağlantı yolu bulut konteynerinden (TCP engelli) çalıştırılamadı. Üretimde etkinleştirmek için Vercel'de `DATABASE_URL` ortam değişkeni gerekir; yoksa her instance kendi geçici SQLite dosyasını kullanır. Sürümlü migrasyonlar (`schema_migrations`, alembic yerine 60 satırlık çözüm: tek tablo için yeterli), token korumalı kalite mühendisi ucu, `/spc` p-grafiği, CI'da Postgres servisi ve psycopg yolunun gerçek Postgres'te doğrulanması tamam.
- Historian: Neon Postgres (deploy) / SQLite (yerel); şema `alembic` ile
- Düşük güvenli plakalar için inceleme kuyruğu UI'ı (onayla / düzelt)
- Düzeltilen etiketlerin yeniden eğitim adayı olarak dışa aktarımı
- Operatör / kalite mühendisi rolleri
- Vardiya bazlı SPC grafikleri; Excel rapor bu veriden

## Faz 4: RAG'i kusur akışına bağlama (GPU/model gerektirmeyen kısmı tamam)

Hazır: inceleme kuyruğunda kusur sınıfına göre bilgi tabanı pasajları (`/api/defect-knowledge`, TF-IDF; model gerektirmez). Kalan, elle ve model erişimi gerektiren işler: 30-50 gerçekçi soruluk ikinci eval seti, insan etiketli "cevap doğru mu" alt kümesi, açık lisanslı SOP benzeri kaynaklarla bilgi tabanı genişletme. Bilinen sınır: KB'deki *Crazing* makalesi polimerlerle ilgili, çelik yüzey çatlağı değil.
- Sınıflandırıcı çıktısından RAG'e: kusurun nedenleri ve düzeltici eylemler
- Açık lisanslı kaynaklarla bilgi tabanı genişletme (lisans kaydıyla)
- 30-50 elle yazılmış gerçekçi soruluk ikinci eval seti
- İnsan etiketli küçük "cevap doğru mu" alt kümesi

## Faz 5: Canlı akışa doğru (temeli tamam)

Hazır: `FrameSource` (simülatör / video döngüsü), `analysis/bench_frames.py` (p50/p95, fps), `Dockerfile` + `docker-compose.yml`. Kalan: ONNX çıkarım gecikmesini aynı ölçüme katmak (gerçek model gerekir).
- `FrameSource` soyutlaması: simülatör / video dosyası
- NEU-CLS video döngüsüyle gerçek inference hattı; gecikme (p50/p95) ve verim ölçümü
- `Dockerfile` + `docker-compose` (app + Postgres)

## Faz 6: Yayın (kod tarafı tamam)

Hazır: `CHANGELOG.md`, README güncel. Mimari diyagram (Mermaid) ve `0.2.0` sürüm notu hazır. Kalan, koddan bağımsız: sürüm etiketi ve güncel demo videosu (elle), gerçek NEU-CLS CNN sonuçları (GPU çalıştırması sonrası `analysis/update_docs_from_dl.py`).
- Sürüm etiketleri, `CHANGELOG.md`, README'nin kısaltılması, mimari diyagram, güncel demo videosu

## Riskler
| Risk | Önlem |
|------|-------|
| Vercel boyut/soğuk başlangıç sınırı | ONNX + küçük model; ağır işler Docker'da |
| Derin model RF'ten az farkla iyi çıkabilir | CV ve güven aralığıyla dürüstçe raporla |
| Yüklenen görüntülerin gizliliği | Saklama yok veya kısa TTL; kullanıcıya belirt |
| Neon kaynaklarının başka projelerle karışması | Ayrı proje, ayrı rol, ayrı `DATABASE_URL`; adlandırma `visual-qc-*` |
