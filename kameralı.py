from PIL import Image   
import cv2  #pip install cv2
import imutils #pip install imutils
import pytesseract  #pip install pytesseract kodlarını terminale girerek eklenti yüklemesi yapmamız gerekli
from pytesseract.pytesseract import image_to_alto_xml, image_to_string

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe" #TeseractOCR windowsa kurduktan sonra nerede olduğunu tanıtmamız gerekli 
plaka_cascade=cv2.CascadeClassifier(r"plaka.xml") #Plakanın kameradan tanınması için gerekli xml verilerini ekliyoruz
cap = cv2.VideoCapture(0) #cameranın çalışması için gerekli kod
i=0
image=None
while cap.isOpened(): #Eğer kamera açıksa sürekli tekrarlıyor
    ret,frame= cap.read() #kameradan gelen görüntüyü okuyor kare kare
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #gelen görüntünün gray filtreli haline dönüştürüyor

    faces = plaka_cascade.detectMultiScale(gray,1.12,4,minSize=(11,52))  #plaka xml verilerine göre detectmultiscale e resmi,ölçeği ve ne kadar boyuttan sonrasını okuması gerektiğini giriyoruz.

    for(x,y,w,h) in faces: #analiz edilen resimde plakanın bulunduğu kısmı dikdörtgen içine alarak o resmi kırpıyor
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),3) #dikdörtgene almak için gerekli çizim kodu
        image = frame[y:y+h, x:x+w]  #resmi belirli ölçilerde kırpıyor
        cv2.imshow('img',image) #cv.imshow ekrana yazdırmak için kullanılıyor.
        cv2.imwrite('plaka'+str(i)+'.jpg',image) #kırptığı resmi kaydediyor
        i+=1
        cv2.waitKey(0) #pencerenin ekranda durması için gerekli 

        
    if cv2.waitKey(1) & 0xFF == ord('q'):  # bu komutta da kamerayı kapatmak istersek q tuşuna basıyoruz.
        break # komuttan çıkış için

    
    
    cv2.imshow('img',frame) #Kamera Görüntüsünü ekrana yazdırıyor

image = imutils.resize(image,height= 300)
cv2.imshow("Orjinal İmage", image) #resmin orjinal halini gösteriyor
cv2.waitKey(0)

cv2.destroyWindow("Orjinal İmage")  #Bir önceki pencereyi öldürüyor
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #resme gray filtresiqq
cv2.imshow("Gray Scale İmage",gray)
cv2.waitKey(0)

cv2.destroyWindow("Gray Scale İmage") #Bir önceki pencereyi öldürüyor
smoth = cv2.bilateralFilter(gray,10,15,15) #
cv2.imshow("Smoother İmage", smoth)
cv2.waitKey(0)

cv2.destroyWindow("Smoother İmage") #Bir önceki pencereyi öldürüyor
edged = cv2.Canny(gray,120,200)
cv2.imshow("Canny Edge",edged)
cv2.waitKey(0)


cnts, new = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

image1 =image.copy()
cv2.destroyWindow("Canny Edge")#Bir önceki pencereyi öldürüyor
cv2.drawContours(image1 , cnts , -1,(0,255,0),2)
cv2.imshow("Canny After Conturing",image1)
cv2.waitKey(10)

cnts=sorted(cnts,key=cv2.contourArea,reverse=True)[:120]
NumberPlateCount= None


image2=image.copy()
cv2.destroyWindow("Canny After Conturing")#Bir önceki pencereyi öldürüyor
cv2.drawContours(image2,cnts,-1,(0,255,0),3) #kenarları çiziyor
cv2.imshow("Top 30 Contours",image2)
cv2.waitKey(0)
crp_img=image
count = 0 
name = 1
for i in cnts:
    perimeter=cv2.arcLength(i,True)  
    approx=cv2.approxPolyDP(i,0.01*perimeter,True) # plakanın köşelerini bulmak için kullanılıyor
    if(len(approx)==4): #4 köşe olduğunda noktaları değişkenlere atıyor ve resmi kurpıyor ve kaydediyor
        NumberPlateCount=approx
        x,y,w,h=cv2.boundingRect(i)
        crp_img=image[y:y+h,x:x+w]
        cv2.imwrite(str(name)+'.png',crp_img)
        count=name
        name +=1
    break


cv2.drawContours(image,[NumberPlateCount],-1,(0,255,0),2)
cv2.imshow("Final İmage",image)
cv2.waitKey(0)

cv2.destroyWindow("Top 30 Contours")
cv2.imshow("Cropped Image",crp_img) #resmin son halini gösteriyor
cv2.waitKey(0)

cv2.destroyWindow("Cropped Image")
text = pytesseract.image_to_string(crp_img,lang='tur') #resimdeki karakterleri okuyor ve ekrana yazdırıyor
print("Number is: ",text)
cv2.waitKey(0)


