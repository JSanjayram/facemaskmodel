import smtplib
import requests
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

class AlertSystem:
    def __init__(self):
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'email': 'your_email@gmail.com',
            'password': 'your_app_password'
        }
        
    def send_email_alert(self, recipient, subject, message):
        """Send email alert for mask violations"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email']
            msg['To'] = recipient
            msg['Subject'] = subject
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email'], self.email_config['password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['email'], recipient, text)
            server.quit()
            
            print(f"Alert sent to {recipient}")
            return True
        except Exception as e:
            print(f"Email alert failed: {e}")
            return False
    
    def send_webhook_alert(self, webhook_url, data):
        """Send webhook alert to external systems"""
        try:
            response = requests.post(webhook_url, json=data, timeout=10)
            if response.status_code == 200:
                print("Webhook alert sent successfully")
                return True
            else:
                print(f"Webhook failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Webhook alert failed: {e}")
            return False
    
    def log_violation(self, violation_data):
        """Log mask violations to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - {violation_data}\n"
        
        with open('mask_violations.log', 'a') as f:
            f.write(log_entry)
    
    def trigger_alert(self, detection_result, location="Unknown"):
        """Trigger appropriate alerts based on detection result"""
        if detection_result['status'] == 'Without Mask':
            # Create alert message
            message = f"""
            MASK VIOLATION DETECTED
            
            Location: {location}
            Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Confidence: {detection_result['confidence']:.1f}%
            
            Immediate action required.
            """
            
            # Log violation
            self.log_violation(f"No mask detected at {location} - Confidence: {detection_result['confidence']:.1f}%")
            
            # Send email alert
            self.send_email_alert(
                "security@company.com",
                "🚨 Mask Violation Alert",
                message
            )
            
            # Send webhook alert
            webhook_data = {
                'alert_type': 'mask_violation',
                'location': location,
                'timestamp': datetime.now().isoformat(),
                'confidence': detection_result['confidence']
            }
            
            self.send_webhook_alert("https://your-webhook-url.com/alerts", webhook_data)

if __name__ == "__main__":
    alert_system = AlertSystem()
    
    # Test alert
    test_result = {'status': 'Without Mask', 'confidence': 95.5}
    alert_system.trigger_alert(test_result, "Main Entrance")