import smtplib
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from email_key import email_id, email_pw, msg_from, msg_to


def send_email_emotion(emotion):
    message = MIMEMultipart()
    message['Subject'] = f'사용자 감정: {emotion}'
    message['From'] = msg_from
    message['To'] = msg_to

    content = f"""
        <html>
        <body>
            <p>사용자 감정 인식 결과입니다</p>
            <p>{emotion}</p>
        </body>
        </html>
    """

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


def send_email_emotion_statistics(before_emotion, before_emotion_prob, after_emotion, after_emotion_prob):
    message = MIMEMultipart()
    message['Subject'] = f'사용자 감정: {before_emotion} -> {after_emotion}'
    message['From'] = msg_from
    message['To'] = msg_to

    content = f"""
    <html>
            <body>
                <p>사용자 감정 인식 결과입니다</p>
                <p>{before_emotion}</p>
                <table>
                    <caption>사용자 감정 인식 결과</caption>
                    <colgroup>
                        <col>
                    </colgroup>
                    <thead>
                        <tr>
                            <th rowspan="2"></th>
                            <!-- <th></th> -->
                            <!-- <th></th> -->
                        </tr>
                        <tr>
                            <!-- <th></th> -->
                            <th width="80" style="background-color:wheat;">화</th>
                            <th width="80" style="background-color:wheat;">짜증</th>
                            <th width="80" style="background-color:wheat;">두려움</th>
                            <th width="80" style="background-color:wheat;">행복</th>
                            <th width="80" style="background-color:wheat;">슬픔</th>
                            <th width="80" style="background-color:wheat;">놀람</th>
                            <th width="80" style="background-color:wheat;">중립</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <th width="80" style="background-color:darkseagreen;">Before</th>
                            <td>10%</td>
                            <td>20%</td>
                            <td>30%</td>
                            <td>5%</td>
                            <td>1%</td>
                            <td>3%</td>
                            <td>36%</td>
                        </tr>
                        <tr>
                            <th width="80" style="background-color:darkseagreen;">After</th>
                            <td>15%</td>
                            <td>23%</td>
                            <td>22%</td>
                            <td>3%</td>
                            <td>1%</td>
                            <td>3%</td>
                            <td>33%</td>
                        </tr>
                        <tr>
                        <th width="80" style="background-color:darkseagreen;">+/-</th>
                            <td style="color:red;">+5%</td>
                            <td style="color:red;">+3%</td>
                            <td style="color:blue;">-8%</td>
                            <td style="color:blue;">-2%</td>
                            <td>-</td>
                            <td>-</td>
                            <td style="color:blue;">-3%</td>
                        </tr>
                    </tbody>
                </table>
            </body>
    </html>
    """

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