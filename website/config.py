#
# Flask Setup
#

from flask import Flask, render_template, request, Blueprint, session, redirect, url_for, jsonify, send_from_directory
#from flask_paginate import Pagination, get_page_parameter, get_page_args
#from werkzeug.utils import secure_filename
import os
#import numpy as np
#import pandas as pd
from datetime import datetime
import json
import csv
from discord_webhook import DiscordWebhook, DiscordEmbed


app = Flask(__name__)
#app.config['SECRET_KEY'] = "xxxxxxxx"

#
# Mail Setup
#
from flask_mail import Mail, Message
app.config['MAIL_SERVER']=''
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = ''
app.config['MAIL_PASSWORD'] = ''
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)

#
# Other Imports
#
import random
import re

#
# Internal Imports
#
#from email_templates import *
from functions import *
