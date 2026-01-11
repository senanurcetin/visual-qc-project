# VisionQC - Fabrika Simülasyon HMI

![Python](https://img.shields.io/badge/python-3.11-blue.svg) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) ![OpenCV](https://img.shields.io/badge/opencv-%235C3EE8.svg?style=for-the-badge&logo=opencv&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Bu proje, bir üretim hattının kalite kontrol (QC) sürecini simüle eden, Flask tabanlı bir web uygulamasıdır. Gerçek zamanlı bir İnsan-Makine Arayüzü (HMI) paneli üzerinden üretim verilerini izleme, sistemi kontrol etme ve detaylı raporlar alabilme imkanı sunar.

![HMI Screenshot](assets/screenshot.png)

## Temel Özellikler

- **Gerçek Zamanlı HMI Paneli:** Üretim durumu, OEE, net kar, ve ürün sayıları gibi kritik performans göstergelerini (KPI) anlık olarak gösterir.
- **Simüle Edilmiş Video Akışı:** OpenCV ile oluşturulan video akışı, üretim bandını görsel olarak simüle eder.
- **Endüstriyel Metrikler:** Standartlara uygun OEE (Kullanılabilirlik, Performans, Kalite) hesaplamaları yapar.
- **Veritabanı Entegrasyonu:** Tüm üretim olaylarını zaman damgasıyla birlikte bir SQLite veritabanına kaydeder.
- **Gelişmiş Raporlama:** Pandas ve XlsxWriter kullanarak, özet dashboard ve grafikler içeren profesyonel Excel raporları oluşturur.
- **API ile Kontrol:** Simülasyonu harici olarak yönetmek için RESTful API uç noktaları sunar.

## Mimarinin Öne Çıkanları

- **Thread-Safe Tasarım:** Eşzamanlı video akışı, API istekleri ve veritabanı işlemlerinin sorunsuz çalışmasını sağlamak için `threading.RLock` kullanılır. Bu, aynı thread'in kilidi birden çok kez almasına izin vererek karmaşık senaryolarda kilitlenmeleri (deadlocks) önler.
- **Non-Blocking I/O:** Veritabanına yazma gibi potansiyel olarak yavaş I/O işlemleri, ana simülasyon ve video oluşturma döngüsünün kilidi dışında gerçekleştirilir. Bu, uygulamanın yanıt vermemesini veya `504 Gateway Timeout` hatası vermesini engeller.
- **Bellek İçi (In-Memory) Rapor Üretimi:** Excel raporları, diske geçici bir dosya yazmak yerine doğrudan bellekteki bir `io.BytesIO` akışına oluşturulur. Bu, I/O'yu azaltır ve rapor oluşturma sürecini hızlandırır.
- **Stateless Mimari Yaklaşımı:** Veritabanı işlemleri için her seferinde yeni bir bağlantı açılıp kapatılır. Bu, her işlemin bağımsız olmasını sağlayarak ve bağlantı havuzlarını yönetme karmaşıklığını ortadan kaldırarak uygulamanın ölçeklenmesini ve bakımını kolaylaştırır.

## Proje Yapısı

```
.
├── main.py              # Ana Flask uygulaması, tüm mantığı içerir
├── requirements.txt     # Proje bağımlılıkları
├── devserver.sh         # Geliştirme sunucusunu başlatan betik
├── vision_qc.db         # Üretim loglarının tutulduğu SQLite veritabanı
└── README.md            # Bu döküman
```

## Kullanılan Teknolojiler

- **Backend:** Python, Flask
- **Video İşleme:** OpenCV (`opencv-python-headless`)
- **Veri Analizi & Raporlama:** Pandas, XlsxWriter
- **Veritabanı:** SQLite
- **Frontend:** HTML, CSS, JavaScript (Chart.js ile)

## Kurulum ve Çalıştırma

1.  **Sanal Ortamı Etkinleştirin:**
    Proje, izolasyon için bir sanal ortam kullanır. İşletim sisteminize uygun komutu terminalde çalıştırın:
    
    *   **Linux/macOS:**
        ```bash
        source .venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        .venv\Scripts\activate
        ```

2.  **Bağımlılıkları Yükleyin:**
    Gerekli tüm Python paketlerini `requirements.txt` dosyasından yükleyin:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Uygulamayı Başlatın:**
    Geliştirme sunucusunu başlatmak için aşağıdaki komutu kullanın:

    * **Linux/macOS:**
        ```bash
        ./devserver.sh
        ```
    * **Windows:**
        ```bash
        python main.py
        ```
    
    Uygulama artık `http://localhost:8080` adresinde çalışıyor olacaktır.

## API Uç Noktaları (Endpoints)

- `GET /`: Ana HMI kullanıcı arayüzünü render eder.
- `GET /video_feed`: Üretim hattının OpenCV ile oluşturulmuş simülasyon video akışını sağlar. (`multipart/x-mixed-replace`)
- `GET /api/data`: Anlık sistem durumu, KPI'lar ve son logları içeren JSON verisini döndürür. HMI paneli bu veriyi periyodik olarak çeker.
- `POST /api/control`: Simülasyonu kontrol etmek için JSON komutları kabul eder.
  - **Body:** `{"command": "COMMAND_NAME"}`
  - **Komutlar:** `START`, `PAUSE`, `RESET`, `ESTOP`, `SIMULATE_FAIL` (bir sonraki ürünü kasıtlı olarak hatalı üretir).
- `GET /api/export_report`: Veritabanındaki tüm üretim geçmişini içeren, formatlanmış bir Excel (.xlsx) raporu oluşturur ve indirme linki sağlar.

## Gelecek Planları & Geliştirmeler

- [ ] MQTT ile PLC Entegrasyonu
- [ ] YOLOv8 ile Yapay Zeka Destekli Hata Tespiti
- [ ] Docker Konteyner Desteği
- [ ] Bulut Tabanlı Veri İzleme

---
<div align="center">
  <sub>Developed by Senanur Çetin - 2026</sub>
</div>