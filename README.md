# Thermostat Control Simulation

Bu proje, farklı sıcaklık kontrol algoritmalarını karşılaştıran bir simülasyon sunar. On-Off, PID ve Q-Learning algoritmalarının performansını değerlendirmek için geliştirilmiştir. Kullanıcı, çeşitli parametrelerle simülasyonu kişiselleştirebilir ve sonuçları görselleştirebilir.

## Özellikler

- **Algoritma Karşılaştırması**: On-Off, PID ve Q-Learning algoritmalarının performansını karşılaştırın.
- **Dinamik Girdi**: Sıcaklık, güç ve kayıp gibi parametreleri yanıtlamak için ayarlayabilirsiniz.
- **Grafiksel Sonuçlar**: Algoritma performansını ve konfor ile enerji tüketimini gösteren çeşitli grafikler.
- **Kullanıcı Dostu Arayüz**: Streamlit tabanlı arayüz ile kolay kullanım.

## Parametreler
Başlangıç Sıcaklığı: Oda sıcaklığının başlangıç değeri.
Termostat Ayarı: İstenilen sıcaklık değeri.
Isıtıcı Gücü: Isıtıcının sıcaklığı artırma hızı.
Temel Isı Kaybı: Sıcaklık kaybının oranı.
Simülasyon Süresi: Simülasyonun süresi (dakika).
PID Parametreleri: PID kontrolü için Kp, Ki ve Kd değerleri.
Q-Learning Parametreleri: Eğitim epizotları, öğrenme oranı, indirim faktörü, keşif oranı.

## Sonuçlar
Simülasyon çalıştırıldığında:

Grafik 1: Sıcaklık kontrol algoritmalarının oda sıcaklıkları üzerindeki etkisi.
Grafik 2: Her algoritmanın overshoot ve undershoot değerlerinin karşılaştırılması.
Grafik 3: Toplam overshoot ve undershoot değerlerinin karşılaştırması.
Grafik 4: Dış ortam sıcaklığı değişimi.

## Katkıda Bulunma
Herhangi bir katkıda bulunmak isterseniz, lütfen bir pull request oluşturun veya bir sorun bildirin.