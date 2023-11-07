import smtplib
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from email_key import msg_from, msg_to, email_id, email_pw


def send_email(emotion):
    message = MIMEMultipart()
    message['Subject'] = '사용자 감정: {emotion}'.format(emotion = emotion)
    message['From'] = msg_from
    message['To'] = msg_to

    content = """
        <html>
        <body>
            <p>사용자 감정 인식 결과입니다</p>
            <p>{emotion}</p>
        </body>
        </html>
    """.format(emotion = emotion)

    mimetext = MIMEText(content,'html')
    message.attach(mimetext)

    try:
        server = smtplib.SMTP('smtp.naver.com',587)
        server.ehlo()
        server.starttls()
        server.login(email_id,email_pw)
        server.sendmail(message['From'],message['To'],message.as_string())
        server.quit()
    except:
        print("[error] send email fail")
        return False
    return True