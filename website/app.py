from config import *
from api_routes import api_routes
main_routes = Blueprint('main_routes', __name__, template_folder='templates')

app.register_blueprint(api_routes)

@app.route('/')
def index():
    date = getDate()
    data = getArticles()
    return render_template('index.html', date=date, data=data)

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = SearchForm()
    if request.method == 'POST':
        return redirect((url_for('search_results', query=form.search.data)))  # or what you want
    return render_template('index.html', form=form)

@app.route('/about')
def about():
    date = getDate()
    return render_template('about.html', date=date)

if __name__ == "__main__":
    app.run(debug=True)
