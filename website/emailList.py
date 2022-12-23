#email
from config import *

# Function which sends an email template populated with content of JSON structure
def create_app(**kwargs):
    # Initialising all credentials for mail server
    app2 = Flask(__name__)
    app2.config['MAIL_SERVER']='SERVER'
    app2.config['MAIL_PORT'] = 465
    app2.config['MAIL_USERNAME'] = 'USERNAME'
    app2.config['MAIL_PASSWORD'] = 'PASSWORD'
    app2.config['MAIL_USE_TLS'] = False
    app2.config['MAIL_USE_SSL'] = True
    mail = Mail(app2)

    # Allowing python flask operations to operate with the absence of a request 
    with app2.app_context():
        # Python flask for bulk email sendings
        with mail.connect() as conn:
            with open("emails.csv", 'r') as reader:
                # For each email in the CSV file send an email containing a title and the HTML template from Mail server
                for row in reader.readlines()[1:]:
                    try:
                        msg = Message("Twitter Rewind Catchup",recipients=[row.strip()],sender="USERNAME")
                        msg.html = render_template('emailFormat.html', **kwargs)
                        conn.send(msg)
                    except:
                        print("Email skipped")
    return app2

# Retrieve current date
date = datetime.now()
date = date.strftime("%d/%m/%y")

# Load JSON file content into JSON object and pass to function
f = open('output.json')
data = json.load(f)
create_app(date = date, data = data)

f.close()
