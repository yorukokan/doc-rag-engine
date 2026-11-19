# DNS ve Ağ Temelleri

## DNS Nedir ve Nasıl Çalışır?
DNS (Domain Name System), insanların okuyabildiği alan adlarını (örneğin google.com) makinelerin anladığı IP adreslerine çeviren dağıtık bir sistemdir. Tarayıcıya bir adres yazıldığında önce yerel önbellek (cache), ardından recursive resolver, root sunucular, TLD sunucuları ve yetkili (authoritative) isim sunucuları sırayla sorgulanır.

## Temel DNS Kayıt Türleri
A kaydı bir alan adını IPv4 adresine, AAAA kaydı ise IPv6 adresine yönlendirir. CNAME kaydı bir alan adını başka bir alan adına takma ad (alias) olarak bağlar. MX kayıtları e-posta sunucularını belirlerken, TXT kayıtları genellikle SPF ve DKIM gibi güvenlik doğrulamaları için kullanılır.

## Wildcard DNS Kayıtları
Wildcard DNS (*.ornek.com), tanımlanmamış tüm alt alan adlarının tek bir IP adresine veya hedefe yönlendirilmesini sağlar. Özellikle çoklu kiracılı (multi-tenant) uygulamalarda ve geliştirme ortamlarında her alt alan adı için tek tek kayıt açma yükünü ortadan kaldırır. Ancak yanlış yapılandırıldığında güvenlik açıklarına veya beklenmedik trafik yönlendirmelerine yol açabilir.

## TTL (Time to Live) Kavramı
TTL, bir DNS kaydının çözümleyici (resolver) önbelleğinde kaç saniye boyunca saklanacağını belirler. Düşük TTL değerleri (örneğin 300 saniye) sunucu taşıma işlemlerinde değişikliklerin hızlı yayılmasını sağlar ama DNS sunucusundaki sorgu yükünü artırır. Yüksek TTL değerleri ise performansı artırır ve yükü azaltır.
