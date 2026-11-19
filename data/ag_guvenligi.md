# Ağ Güvenliği ve Temel Kavramlar

## Firewall (Güvenlik Duvarı) ve NAT
Firewall, gelen ve giden ağ trafiğini belirlenen güvenlik kurallarına göre filtreleyen sistemdir. Durum bilgili (Stateful) firewall cihazları aktif bağlantıların durumunu takip ederek yalnızca izin verilen oturumların dönüş paketlerine geçiş verir. NAT (Network Address Translation) ise yerel ağdaki özel (private) IP adreslerini internete çıkarken tek bir genel (public) IP adresine dönüştürür.

## VPN ve Şifreli Tünelleme
VPN (Virtual Private Network), genel internet ağı üzerinden iki nokta arasında şifreli bir tünel oluşturarak verilerin güvenli taşınmasını sağlar. Kurumsal ağlarda şubeler arası bağlantı için Site-to-Site VPN, uzaktan çalışanların iç ağa erişimi için Client-to-Site VPN kullanılır. Günümüzde yüksek hızı ve basit yapılandırması nedeniyle WireGuard ve IPsec protokolleri yaygın olarak tercih edilir.

## TLS/SSL ve Sertifika Mantığı
TLS (Transport Layer Security), istemci ile sunucu arasındaki iletişimi şifreleyerek verinin dinlenmesini ve değiştirilmesini önler. Sunucu kimliğini doğrulamak için bir Sertifika Otoritesi (CA) tarafından imzalanmış dijital sertifikalar kullanılır. Sertifikanın süresi dolduğunda veya alan adıyla uyuşmadığında tarayıcılar güvenlik uyarısı vererek bağlantıyı keser.

## DDoS Saldırıları ve Korunma Yöntemleri
DDoS (Distributed Denial of Service), çok sayıda ele geçirilmiş cihazdan tek bir hedefe yoğun trafik göndererek servisi erişilemez hale getirme saldırısıdır. Ağ katmanında SYN Flood gibi saldırılar görülürken, uygulama katmanında (Layer 7) doğrudan HTTP istekleriyle sunucu kaynakları tüketilir. Korunmak için Rate Limiting (istek sınırlama), WAF (Web Application Firewall) ve trafik filtreleme servisleri kullanılır.
