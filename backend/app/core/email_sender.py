import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os

# Configuration (In a real app, use Environment Variables)
# You must set these to make it work!
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "xrayproject.demo@gmail.com" # REPLACE WITH REAL EMAIL
SENDER_PASSWORD ="cugb sstt zeoy yypc"# REPLACE WITH REAL APP PASSWORD

def send_email_with_report(to_email: str, subject: str, body: str, pdf_bytes: bytes, filename: str = "report.pdf"):
    """
    Sends an email with the PDF report attached.
    """
    if "your_email" in SENDER_EMAIL:
        print("WARNING: Email credentials not configured in 'app/core/email_sender.py'. Email will NOT be sent.")
        return False, "Email credentials not configured on server."

    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject

        # Body
        msg.attach(MIMEText(body, 'plain'))

        # Attachment
        part = MIMEApplication(pdf_bytes, Name=filename)
        part['Content-Disposition'] = f'attachment; filename="{filename}"'
        msg.attach(part)

        # Server
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        return True, "Email sent successfully."
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False, str(e)
