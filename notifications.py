import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage


class Notification:
    def __init__(self, sender_email, receiver_email, password):
        self.from_email = sender_email
        self.to_email = receiver_email
        self.password = password
        self.server = None
        self.authenticate()

    def authenticate(self):
        self.server = smtplib.SMTP('smtp.gmail.com: 587')
        self.server.starttls()
        self.server.login(self.from_email, self.password)

    def send_email(self, object_detected=1):
        message = MIMEMultipart()
        message['From'] = self.from_email
        message['To'] = self.to_email
        message['Subject'] = "Intrusion Security Alert"
        message_body = f'''
        <p>ALERT - {object_detected} intruder(s) has been detected !!</p>
        '''
        message.attach(MIMEText(message_body, 'html'))
        
        # Attach all images to the message
        for file in os.listdir("./images"):
            img = open(os.path.join("./images/",file), 'rb').read()
            image = MIMEImage(img, name=file)     
            message.attach(image)
        
        # Send the mail
        self.server.sendmail(self.from_email, self.to_email, message.as_string())
        # with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        #     server.login(self.from_email, self.password)
        #     server.sendmail(self.from_email, self.to_email, message.as_string())
        # print("Email envoyé avec succès !")
    def quit(self):
        self.server.quit()