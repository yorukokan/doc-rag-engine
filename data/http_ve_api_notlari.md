# HTTP Protokolü ve REST API Mimarisi

## HTTP Durum Kodları
HTTP durum kodları sunucunun isteğe verdiği yanıtın sonucunu belirtir. 200 (OK) işlemin başarılı olduğunu, 201 (Created) yeni bir kaynağın oluşturulduğunu gösterir. 400 (Bad Request) istemcinin hatalı veri gönderdiğini, 401 (Unauthorized) kimlik doğrulama gerektiğini, 403 (Forbidden) ise yetkisiz erişimi ifade eder. 500 ve 502 kodları sunucu tarafındaki iç hatalara işaret eder.

## REST API Temel Prensipleri
REST mimarisi durumsuzdur (stateless); yani sunucu istemcinin önceki isteklerini hafızasında tutmaz ve her istek gerekli tüm bilgiyi (örneğin kimlik token'ını) içinde taşır. Veri okumak için GET, yeni kayıt eklemek için POST, mevcut kaydı tamamen güncellemek için PUT, kısmi güncelleme için PATCH ve silmek için DELETE metotları kullanılır.

## Rate Limiting ve Önbellekleme (Caching)
API sunucularını aşırı yükten korumak için istemcilere belirli bir sürede atabilecekleri maksimum istek sınırı (Rate Limiting) konur; sınır aşıldığında 429 (Too Many Requests) kodu döner. Sık değişmeyen verilerde veritabanı yükünü azaltmak için Redis gibi bellek içi önbellekler kullanılır ve yanıt başlıklarına `Cache-Control` eklenir.

## JWT (JSON Web Token) ile Kimlik Doğrulama
JWT, kullanıcı doğrulandıktan sonra sunucu tarafından üretilen ve istemciye verilen imzalı bir belirteçtir. Header, Payload ve Signature olmak üzere üç bölümden oluşur. İstemci sonraki her API isteğinde bu token'ı `Authorization: Bearer <token>` başlığıyla gönderir ve sunucu veritabanına tekrar sormadan imzayı doğrulayarak kullanıcıyı tanır.
