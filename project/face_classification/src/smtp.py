import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from string import Template

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
    message['Subject'] = f'사용자 감정 변화 결과입니다.'
    message['From'] = msg_from
    message['To'] = msg_to
    
    before_data = []
    after_data = []
    increase = []
    increase_color = []
    
    for i in range(len(before_emotion_prob)):
        before_data.append(int(before_emotion_prob[i] // 0.01))
    for i in range(len(after_emotion_prob)):
        after_data.append(int(after_emotion_prob[i] // 0.01))
    for i in range(len(before_emotion_prob)):
        temp = after_data[i] - before_data[i]
        if temp > 0:
            temp = '+' + str(temp) + '%'
            increase.append(temp)
            increase_color.append('red')
        elif temp < 0:
            temp = str(temp) + '%'
            increase.append(temp)
            increase_color.append('blue')
        else:
            temp = '-'
            increase.append(temp)
            increase_color.append('black')
        

    content = f"""
    <html>
            <body>
                <p>{before_emotion} -> {after_emotion}</p>
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
                            <td width="80">{before_data[0]}%</td>
                            <td width="80">{before_data[1]}%</td>
                            <td width="80">{before_data[2]}%</td>
                            <td width="80">{before_data[3]}%</td>
                            <td width="80">{before_data[4]}%</td>
                            <td width="80">{before_data[5]}%</td>
                            <td width="80">{before_data[6]}%</td>
                        </tr>
                        <tr>
                            <th width="80" style="background-color:darkseagreen;">After</th>
                            <td width="80">{after_data[0]}%</td>
                            <td width="80">{after_data[1]}%</td>
                            <td width="80">{after_data[2]}%</td>
                            <td width="80">{after_data[3]}%</td>
                            <td width="80">{after_data[4]}%</td>
                            <td width="80">{after_data[5]}%</td>
                            <td width="80">{after_data[6]}%</td>
                        </tr>
                        <tr>
                        <th width="80" style="background-color:darkseagreen;">+/-</th>
                            <td style="color:{increase_color[0]};">{increase[0]}</td>
                            <td style="color:{increase_color[1]};">{increase[1]}</td>
                            <td style="color:{increase_color[2]};">{increase[2]}</td>
                            <td style="color:{increase_color[3]};">{increase[3]}</td>
                            <td style="color:{increase_color[4]};">{increase[4]}</td>
                            <td style="color:{increase_color[5]};">{increase[5]}</td>
                            <td style="color:{increase_color[6]};">{increase[6]}</td>
                        </tr>
                    </tbody>
                </table>
            </body>
    </html>
    """

    mimetext = MIMEText(content,'html')
    message.attach(mimetext)

    assert os.path.isfile("../images/before_image.png"), 'image file does not exist.'        
    with open("../images/before_image.png", 'rb') as img_file:
        mime_img = MIMEImage(img_file.read())
        mime_img.add_header('Content-ID', '<' + 'before_image' + '>')
    message.attach(mime_img)

    assert os.path.isfile("../images/after_image.png"), 'image file does not exist.'        
    with open("../images/after_image.png", 'rb') as img_file:
        mime_img = MIMEImage(img_file.read())
        mime_img.add_header('Content-ID', '<' + 'after_image' + '>')
    message.attach(mime_img)

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