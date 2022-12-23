from config import *

# Get current date from system
def getDate():
    date = datetime.now()
    return date.strftime("%d/%m/%y")

# Return the contents of output.json as a JSON structure
def getArticles():
    f = open('output.json')
    data = json.load(f)
    f.close()
    return data

# Regiuster an email by storing it in emails.csv after validation and verificaiton
def register(email):
    #Check format using light regex
    mailformat = "[^@]+@[^@]+\.[^@]+"
    if not re.search(mailformat,email):   
        return {"message": "Email Invalid", "success": False}
    else:   
        #Check email not already in database
        with open("emails.csv", 'r') as reader:
            for row in reader.readlines()[0:]:
                if row.strip() == email:
                    return {"message": "Email all ready exists", "success": False}
        # Add email
        with open("emails.csv", "a") as fo:
            fo.write(email + "\n")

        return {"message": "Account registered", "success": True}