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

## Faz 1: Sağlamlaştırma
- `main.py` modüllere bölünür (routes, export, frame)
- `coverage` ölçümü ve CI eşiği; `ruff` + `mypy`; `pre-commit`; Dependabot
- README/case-study sayıları JSON'dan üretilir

## Faz 2: Gerçek model, gerçek görüntü
- ResNet18 / EfficientNet-B0 fine-tune; aynı 80/20 holdout ve metrik tablosuyla Random Forest ile karşılaştırma
- Temperature scaling, reliability diagram, kalibre inceleme kuyruğu
- 5-fold stratified CV ve bootstrap güven aralıkları
- Grad-CAM; scratches↔inclusion karışıklığının incelenmesi
- Dashboard'da prosedürel doku yerine gerçek NEU-CLS örnekleri
- ONNX export ve `POST /api/classify` (yükle, sınıflandır, güven + Grad-CAM)

Çıkış kriteri: karşılaştırma tablosu README'de; CV ve kalibrasyon sonuçları belgelenmiş.

## Faz 3: Kalıcılık ve operatör akışı
- Historian: Neon Postgres (deploy) / SQLite (yerel); şema `alembic` ile
- Düşük güvenli plakalar için inceleme kuyruğu UI'ı (onayla / düzelt)
- Düzeltilen etiketlerin yeniden eğitim adayı olarak dışa aktarımı
- Operatör / kalite mühendisi rolleri
- Vardiya bazlı SPC grafikleri; Excel rapor bu veriden

## Faz 4: RAG'i kusur akışına bağlama
- Sınıflandırıcı çıktısından RAG'e: kusurun nedenleri ve düzeltici eylemler
- Açık lisanslı kaynaklarla bilgi tabanı genişletme (lisans kaydıyla)
- 30-50 elle yazılmış gerçekçi soruluk ikinci eval seti
- İnsan etiketli küçük "cevap doğru mu" alt kümesi

## Faz 5: Canlı akışa doğru (opsiyonel)
- `FrameSource` soyutlaması: simülatör / video dosyası
- NEU-CLS video döngüsüyle gerçek inference hattı; gecikme (p50/p95) ve verim ölçümü
- `Dockerfile` + `docker-compose` (app + Postgres)

## Faz 6: Yayın
- Sürüm etiketleri, `CHANGELOG.md`, README'nin kısaltılması, mimari diyagram, güncel demo videosu

## Riskler
| Risk | Önlem |
|------|-------|
| Vercel boyut/soğuk başlangıç sınırı | ONNX + küçük model; ağır işler Docker'da |
| Derin model RF'ten az farkla iyi çıkabilir | CV ve güven aralığıyla dürüstçe raporla |
| Yüklenen görüntülerin gizliliği | Saklama yok veya kısa TTL; kullanıcıya belirt |
| Neon kaynaklarının başka projelerle karışması | Ayrı proje, ayrı rol, ayrı `DATABASE_URL`; adlandırma `visual-qc-*` |
