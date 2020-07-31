# import libraries
import cv2
import imageio

#load cascade

"""
Burada opencvnin önceden kendisinin eğittiklerini alıyoruz.
Yani bir eğitim işlemi yok.
"""
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')

# yüz tanıma fonksiyonu
def detect(frame):

    # renkleri griye değiştirme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # resim ölçeklendirme ve karenin en yakın komşularını seçme
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # her yüz için çizilecek olan dikdörtgen fonksiyonu
    for(x, y, w, h) in faces:

        # dikdörtgen koordinatları ve kenar ayarları
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        #göz tespiti
        gray_face = gray[y:y+h,x:x+w]
        color_face = frame[y:y + h, x:x + w]
        # yüz için yapılan ölçeklendirmenin aynısı farklı ölçekte ve farklı komşu sayısı
        eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 3)

        # göz için dikdörtgen çizdirme
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(color_face, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    return frame

# videoyu işlemi

# videoyu okuma
reader = imageio.get_reader('1.mp4')
# reader'dan alınan frame per second
fps = reader.get_meta_data()['fps']
# oluşturulacak olan video
writer= imageio.get_writer('output.mp4', fps=fps)

# videoda yüz tanıma
for i, frame in enumerate(reader):
    frame = detect(frame)
    writer.append_data(frame)

    # frame takibi
    print(i)

writer.close()