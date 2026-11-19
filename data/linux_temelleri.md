# Linux Sistem Yönetimi Temelleri

## Dosya İzinleri (chmod ve chown)
Linux sistemlerde her dosya ve dizinin okuma (r=4), yazma (w=2) ve çalıştırma (x=1) izinleri vardır. Bu izinler dosya sahibi (user), grup (group) ve diğer kullanıcılar (others) için ayrı ayrı tanımlanır. Örneğin `chmod 750` komutu dosya sahibine tam yetki verirken, gruba sadece okuma ve çalıştırma izni verir, diğerlerini tamamen engeller. Sahiplik değişimi için ise `chown kullanici:grup dosya` komutu kullanılır.

## Servis Yönetimi (systemd)
Modern Linux dağıtımlarında arka plan servislerini (daemon) yönetmek için `systemd` ve `systemctl` aracı kullanılır. Bir servisi anlık başlatmak için `systemctl start servis_adi`, sistem açılışında otomatik başlamasını sağlamak için ise `systemctl enable servis_adi` çalıştırılır. Servisin hata loglarını incelemek için `journalctl -u servis_adi -f` komutu canlı takip sağlar.

## Süreç ve Kaynak İzleme
Sistemdeki anlık CPU ve RAM tüketimini görmek için `top` veya daha gelişmiş arayüze sahip `htop` kullanılır. Disk doluluk oranlarını bölümlere göre incelemek için `df -h`, belirli bir klasörün ne kadar yer kapladığını bulmak için `du -sh klasor_adi` komutu tercih edilir. Yanıt vermeyen bir süreci sonlandırmak için önce `ps aux` ile PID numarası bulunur, ardından `kill -9 PID` ile kapatılır.

## SSH Güvenliği ve Anahtar Kullanımı
Uzak sunuculara güvenli bağlanmak için SSH protokolü kullanılır. Parola ile giriş yerine `ssh-keygen` ile üretilen açık/özel anahtar çifti (public/private key) kullanmak kaba kuvvet (brute-force) saldırılarını önler. Sunucudaki `/etc/ssh/sshd_config` dosyasından `PermitRootLogin no` ve `PasswordAuthentication no` ayarlarını yapmak temel güvenlik sıkılaştırma adımıdır.
